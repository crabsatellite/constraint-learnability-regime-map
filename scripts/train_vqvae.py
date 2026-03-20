"""
Train 3D VQ-VAE on Minecraft builds.

Phase A of the overnight pipeline:
  - Trains VQ-VAE to reconstruct 32³ voxel grids
  - Saves checkpoints every 30 minutes
  - After training, encodes all builds to latent indices
"""

import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.vqvae import VQVAE3D
from scripts.dataset_dense import DenseVoxelDataset


def weighted_ce_loss(logits, targets, air_weight=0.05):
    """
    Cross-entropy with downweighted air tokens.
    Air (token 0) dominates ~97% of voxels -- downweight to focus on building blocks.
    """
    B, V, D, H, W = logits.shape
    logits_flat = logits.permute(0, 2, 3, 4, 1).reshape(-1, V)  # (B*D*H*W, V)
    targets_flat = targets.reshape(-1)  # (B*D*H*W,)

    # Per-element CE loss
    loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')

    # Downweight air tokens
    weight = torch.ones_like(loss)
    weight[targets_flat == 0] = air_weight

    return (loss * weight).sum() / weight.sum()


def compute_accuracy(logits, targets):
    """Compute per-voxel accuracy, separately for air and non-air."""
    preds = logits.argmax(dim=1)  # (B, D, H, W)
    correct = (preds == targets)

    air_mask = (targets == 0)
    non_air_mask = ~air_mask

    total_acc = correct.float().mean().item()
    air_acc = correct[air_mask].float().mean().item() if air_mask.any() else 0.0
    non_air_acc = correct[non_air_mask].float().mean().item() if non_air_mask.any() else 0.0

    return total_acc, air_acc, non_air_acc


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Training VQ-VAE for {args.steps} steps")

    # Dataset
    dataset = DenseVoxelDataset(
        args.data_dir,
        max_dim=32,
        min_blocks=20,
        augment=True,
    )
    print(f"Dataset: {len(dataset)} builds")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=True,
        drop_last=True,
    )

    # Vocab size from dataset's remapping (top-512 + air = 513)
    vocab_size = dataset.vocab_size
    print(f"Vocab size: {vocab_size} (remapped top-512 blocks)")

    # Model
    model = VQVAE3D(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        code_dim=args.code_dim,
        num_codes=args.num_codes,
        n_downsample=args.n_downsample,
    ).to(device)
    print(f"Latent grid: {model.latent_size}^3 = {model.latent_size**3} tokens (n_downsample={args.n_downsample})")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # Learning rate scheduler: linear warmup + cosine decay
    def lr_schedule(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # AMP scaler for mixed precision
    scaler = torch.amp.GradScaler('cuda')

    # Checkpoint dir
    ckpt_dir = PROJECT_ROOT / "checkpoints" / "vqvae"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint if available
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_step = ckpt['step']
        print(f"Resumed from {args.resume} at step {start_step}")

    # Training loop
    model.train()
    data_iter = iter(loader)
    step = start_step
    log_interval = 50
    save_interval = args.save_every
    revive_interval = 1000

    start_time = time.time()
    last_save_time = start_time
    running_loss = 0
    running_recon = 0
    running_vq = 0
    running_perp = 0
    running_acc = 0
    running_noair_acc = 0

    print(f"\nStarting training...")
    print(f"Batch size: {args.batch_size}, Steps: {args.steps}")
    print(f"Checkpoints saved to: {ckpt_dir}")
    print("-" * 80)

    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        x = batch.to(device)  # (B, 32, 32, 32) int64

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits, vq_loss, perplexity, indices = model(x)
            recon_loss = weighted_ce_loss(logits, x, air_weight=args.air_weight)
            loss = recon_loss + vq_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Metrics
        with torch.no_grad():
            total_acc, air_acc, noair_acc = compute_accuracy(logits, x)

        running_loss += loss.item()
        running_recon += recon_loss.item()
        running_vq += vq_loss.item()
        running_perp += perplexity.item()
        running_acc += total_acc
        running_noair_acc += noair_acc

        step += 1

        # Dead code revival
        if step % revive_interval == 0:
            z = model.encode(x)
            z_flat = z.permute(0, 2, 3, 4, 1).reshape(-1, args.code_dim)
            n_revived = model.quantizer.revive_dead_codes(z_flat)
            if n_revived > 0:
                print(f"  [Step {step}] Revived {n_revived} dead codes")

        # Logging
        if step % log_interval == 0:
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            eta_sec = (args.steps - step) / steps_per_sec

            avg_loss = running_loss / log_interval
            avg_recon = running_recon / log_interval
            avg_vq = running_vq / log_interval
            avg_perp = running_perp / log_interval
            avg_acc = running_acc / log_interval
            avg_noair = running_noair_acc / log_interval
            lr = scheduler.get_last_lr()[0]

            print(f"[Step {step:>6d}/{args.steps}] "
                  f"loss={avg_loss:.4f} (recon={avg_recon:.4f} vq={avg_vq:.4f}) "
                  f"perp={avg_perp:.0f}/{args.num_codes} "
                  f"acc={avg_acc:.3f} noair={avg_noair:.3f} "
                  f"lr={lr:.2e} "
                  f"speed={steps_per_sec:.1f}it/s "
                  f"ETA={eta_sec/60:.0f}min")

            running_loss = 0
            running_recon = 0
            running_vq = 0
            running_perp = 0
            running_acc = 0
            running_noair_acc = 0

        # Save checkpoint
        if step % save_interval == 0 or step == args.steps:
            ckpt_path = ckpt_dir / f"vqvae_step{step}.pt"
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'vocab_size': vocab_size,
                'args': vars(args),
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time/60:.1f} minutes")

    # === Phase A.2: Encode all builds to latent indices ===
    print("\n" + "=" * 80)
    print("Encoding all builds to latent indices...")

    model.eval()
    eval_dataset = DenseVoxelDataset(
        args.data_dir,
        max_dim=32,
        min_blocks=20,
        augment=False,  # No augmentation for encoding
    )
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size * 2,
                             shuffle=False, num_workers=0, pin_memory=True)

    all_indices = []
    total_recon_acc = 0
    total_noair_acc = 0
    n_batches = 0

    with torch.no_grad():
        for batch in eval_loader:
            x = batch.to(device)
            indices = model.encode_to_indices(x)  # (B, 8, 8, 8)
            all_indices.append(indices.cpu().numpy())

            # Also compute reconstruction accuracy
            logits = model.decode_from_indices(indices)
            _, _, noair_acc = compute_accuracy(logits, x)
            total_noair_acc += noair_acc
            n_batches += 1

    all_indices = np.concatenate(all_indices, axis=0)  # (N, 8, 8, 8)
    avg_noair_recon = total_noair_acc / n_batches

    latent_path = PROJECT_ROOT / "data" / "processed" / "latent_codes.npz"
    np.savez_compressed(latent_path, indices=all_indices)
    print(f"Saved {all_indices.shape[0]} latent code arrays to {latent_path}")
    print(f"Shape: {all_indices.shape}")
    print(f"Codebook usage: {len(np.unique(all_indices))} / {args.num_codes} codes used")
    print(f"Non-air reconstruction accuracy: {avg_noair_recon:.3f}")

    # Save config for AR training
    config = {
        'vocab_size': vocab_size,
        'num_codes': args.num_codes,
        'code_dim': args.code_dim,
        'embed_dim': args.embed_dim,
        'hidden_dim': args.hidden_dim,
        'vqvae_checkpoint': str(ckpt_dir / f"vqvae_step{args.steps}.pt"),
        'latent_path': str(latent_path),
        'n_builds': all_indices.shape[0],
        'noair_recon_accuracy': avg_noair_recon,
        'codebook_usage': int(len(np.unique(all_indices))),
    }
    config_path = PROJECT_ROOT / "checkpoints" / "vqvae" / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved: {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default=str(PROJECT_ROOT / "data" / "processed"))
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--steps', type=int, default=150000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--code_dim', type=int, default=128)
    parser.add_argument('--num_codes', type=int, default=1024)
    parser.add_argument('--air_weight', type=float, default=0.05)
    parser.add_argument('--save_every', type=int, default=10000)
    parser.add_argument('--n_downsample', type=int, default=3,
                        help='Number of downsample stages (2=8³ latent, 3=4³ latent)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    train(args)
