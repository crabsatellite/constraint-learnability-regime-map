"""
Train AR Transformer on VQ-VAE latent codes.

Phase B of the overnight pipeline:
  - Loads pre-encoded latent indices from VQ-VAE
  - Trains autoregressive transformer to predict code sequences
  - Saves checkpoints and generates samples periodically
"""

import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.ar_transformer import ARTransformer3D


class LatentCodeDataset(Dataset):
    """Dataset of VQ-VAE latent codes for AR training."""

    def __init__(self, latent_path, augment=True):
        data = np.load(latent_path)
        self.indices = data['indices']  # (N, 8, 8, 8)
        self.augment = augment
        print(f"Loaded {len(self.indices)} latent code arrays")
        print(f"Codebook range: {self.indices.min()} - {self.indices.max()}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        grid = self.indices[idx].copy()  # (8, 8, 8)

        # Data augmentation: random rotation in XZ plane
        if self.augment:
            k = np.random.randint(0, 4)
            if k > 0:
                grid = np.rot90(grid, k=k, axes=(0, 2)).copy()
            if np.random.random() > 0.5:
                grid = np.flip(grid, axis=0).copy()

        # Flatten to sequence in raster order: x-major (x changes slowest)
        seq = grid.reshape(-1)  # (512,)
        return torch.tensor(seq, dtype=torch.long)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load VQ-VAE config
    vqvae_config_path = PROJECT_ROOT / "checkpoints" / "vqvae" / "config.json"
    with open(vqvae_config_path, 'r') as f:
        vqvae_config = json.load(f)

    num_codes = vqvae_config['num_codes']
    latent_path = vqvae_config['latent_path']
    print(f"Using {num_codes} codebook entries")
    print(f"VQ-VAE non-air recon accuracy: {vqvae_config.get('noair_recon_accuracy', 'N/A')}")

    # Dataset
    dataset = LatentCodeDataset(latent_path, augment=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    # Determine latent grid size from encoded data
    sample = dataset.indices[0]  # (D, H, W)
    grid_size = sample.shape[0]  # 4 for 4³, 8 for 8³
    seq_len = grid_size ** 3
    print(f"Latent grid: {grid_size}^3 = {seq_len} tokens")

    # Model
    model = ARTransformer3D(
        num_codes=num_codes,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        max_seq_len=seq_len,
        num_tags=0,
        grid_size=grid_size,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"AR Transformer parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # LR schedule: linear warmup + cosine decay
    def lr_schedule(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * progress))  # min lr = 10% of max

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # AMP
    scaler = torch.amp.GradScaler('cuda')

    # Checkpoint dir
    ckpt_dir = PROJECT_ROOT / "checkpoints" / "ar"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    model.train()
    data_iter = iter(loader)
    step = 0
    log_interval = 50
    save_interval = args.save_every

    start_time = time.time()
    running_loss = 0
    running_acc = 0
    running_top5_acc = 0

    print(f"\nStarting AR training...")
    print(f"Batch size: {args.batch_size}, Steps: {args.steps}")
    print(f"Sequence length: {seq_len} ({grid_size}x{grid_size}x{grid_size})")
    print(f"Checkpoints saved to: {ckpt_dir}")
    print("-" * 80)

    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        codes = batch.to(device)  # (B, 512)

        # Teacher-forced AR: input = [BOS, c0, c1, ..., c510], target = [c0, c1, ..., c511]
        bos = torch.full((codes.shape[0], 1), model.bos_token_id,
                         dtype=torch.long, device=device)
        input_seq = torch.cat([bos, codes[:, :-1]], dim=1)  # (B, 512)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(input_seq)  # (B, 512, num_codes)
            loss = F.cross_entropy(
                logits.reshape(-1, num_codes),
                codes.reshape(-1),
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Metrics
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == codes).float().mean().item()
            # Top-5 accuracy
            top5 = logits.topk(5, dim=-1).indices
            top5_match = (top5 == codes.unsqueeze(-1)).any(dim=-1).float().mean().item()

        running_loss += loss.item()
        running_acc += acc
        running_top5_acc += top5_match
        step += 1

        if step % log_interval == 0:
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            eta_sec = (args.steps - step) / steps_per_sec

            avg_loss = running_loss / log_interval
            avg_acc = running_acc / log_interval
            avg_top5 = running_top5_acc / log_interval
            lr = scheduler.get_last_lr()[0]

            print(f"[Step {step:>6d}/{args.steps}] "
                  f"loss={avg_loss:.4f} "
                  f"acc={avg_acc:.3f} top5={avg_top5:.3f} "
                  f"ppl={np.exp(avg_loss):.1f} "
                  f"lr={lr:.2e} "
                  f"speed={steps_per_sec:.1f}it/s "
                  f"ETA={eta_sec/60:.0f}min")

            running_loss = 0
            running_acc = 0
            running_top5_acc = 0

        if step % save_interval == 0 or step == args.steps:
            ckpt_path = ckpt_dir / f"ar_step{step}.pt"
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'args': vars(args),
                'num_codes': num_codes,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    total_time = time.time() - start_time
    print(f"\nAR training complete in {total_time/60:.1f} minutes")
    print(f"Final checkpoint: {ckpt_dir / f'ar_step{args.steps}.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--save_every', type=int, default=5000)
    args = parser.parse_args()
    train(args)
