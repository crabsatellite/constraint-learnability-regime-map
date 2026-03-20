"""
Train AR Transformer with structural constraint conditioning.

Phase B of the v3 pipeline:
  - Loads pre-encoded latent indices from VQ-VAE
  - Loads structural features extracted from training builds
  - Trains conditioned AR transformer with 10% classifier-free guidance dropout
  - Saves checkpoints with conditioning metadata
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


class ConditionedLatentDataset(Dataset):
    """Dataset of VQ-VAE latent codes + structural features for conditioned AR training."""

    def __init__(self, latent_path, features_path, augment=True, uncond_prob=0.1,
                 manifest_path=None, max_dim=32, min_blocks=20):
        import csv

        # Load latent codes
        data = np.load(latent_path)
        self.indices = data['indices']  # (N, 8, 8, 8)

        # Load structural features
        with open(features_path, 'r') as f:
            features_data = json.load(f)

        self.features_dict = features_data['features']
        self.augment = augment
        self.uncond_prob = uncond_prob

        # Feature vocab sizes (for masking with "unconditional" token)
        self.feature_vocabs = ARTransformer3D.STRUCT_FEATURE_VOCABS
        self.feature_keys = [
            'height_bucket', 'size_bucket', 'footprint_bucket',
            'symmetry_flag', 'enclosure_flag', 'complexity_bucket'
        ]

        # Reconstruct the build name ordering from manifest (same filters as DenseVoxelDataset)
        if manifest_path is None:
            manifest_path = Path(latent_path).parent / "manifest.csv"
        with open(manifest_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            entries = list(reader)

        # Apply same filters as DenseVoxelDataset used during encoding
        self.name_list = []
        for m in entries:
            if int(m['non_air_blocks']) < min_blocks:
                continue
            sx = int(m.get('shape_x', 0))
            sy = int(m.get('shape_y', 0))
            sz = int(m.get('shape_z', 0))
            if max(sx, sy, sz) > max_dim:
                continue
            # Use the npz filename as key
            self.name_list.append(Path(m['path']).name)

        print(f"Manifest entries after filtering: {len(self.name_list)}")
        print(f"Latent codes: {len(self.indices)}")
        if len(self.name_list) != len(self.indices):
            print(f"  WARNING: count mismatch! Using min of both.")
            n = min(len(self.name_list), len(self.indices))
            self.name_list = self.name_list[:n]
            self.indices = self.indices[:n]

        # Build features array for fast access
        self._build_features_array()

        print(f"Loaded {len(self.indices)} latent codes with structural features")
        print(f"Feature coverage: {self.n_matched}/{len(self.indices)} builds matched")
        print(f"Unconditional dropout: {uncond_prob:.0%}")

    def _build_features_array(self):
        """Pre-build features array aligned with latent indices."""
        self.features_array = np.zeros((len(self.indices), 6), dtype=np.int64)
        self.n_matched = 0
        uncond_ids = [v for v in self.feature_vocabs]  # mask token = vocab_size

        for idx in range(len(self.indices)):
            name = self.name_list[idx] if idx < len(self.name_list) else None

            # Try exact match, then with .npz extension
            feat = None
            if name:
                feat = self.features_dict.get(name)
                if feat is None:
                    feat = self.features_dict.get(name + '.npz')
                if feat is None:
                    feat = self.features_dict.get(name.replace('.npz', ''))

            if feat is not None:
                for i, key in enumerate(self.feature_keys):
                    self.features_array[idx, i] = feat[key]
                self.n_matched += 1
            else:
                # Use unconditional tokens for unmatched builds
                for i in range(6):
                    self.features_array[idx, i] = uncond_ids[i]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        grid = self.indices[idx].copy()  # (8, 8, 8)
        features = self.features_array[idx].copy()  # (6,)

        # Data augmentation: random rotation in XZ plane
        if self.augment:
            k = np.random.randint(0, 4)
            if k > 0:
                grid = np.rot90(grid, k=k, axes=(0, 2)).copy()
            if np.random.random() > 0.5:
                grid = np.flip(grid, axis=0).copy()

        # Classifier-free guidance dropout: mask ALL features with prob uncond_prob
        if np.random.random() < self.uncond_prob:
            for i in range(6):
                features[i] = self.feature_vocabs[i]  # unconditional mask token

        seq = grid.reshape(-1)  # (512,)
        return torch.tensor(seq, dtype=torch.long), torch.tensor(features, dtype=torch.long)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load VQ-VAE config
    vqvae_config_path = PROJECT_ROOT / "checkpoints" / "vqvae" / "config.json"
    with open(vqvae_config_path, 'r') as f:
        vqvae_config = json.load(f)

    num_codes = vqvae_config['num_codes']
    latent_path = args.latent_path or vqvae_config['latent_path']
    features_path = args.features_path or str(PROJECT_ROOT / "data" / "processed" / "structural_features.json")

    print(f"Using {num_codes} codebook entries")
    print(f"Latent codes: {latent_path}")
    print(f"Features: {features_path}")

    # Dataset
    dataset = ConditionedLatentDataset(
        latent_path, features_path,
        augment=True, uncond_prob=args.uncond_prob,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    # Determine latent grid size
    sample = dataset.indices[0]
    grid_size = sample.shape[0]
    seq_len = grid_size ** 3
    print(f"Latent grid: {grid_size}^3 = {seq_len} tokens")

    # Model with structural conditioning
    model = ARTransformer3D(
        num_codes=num_codes,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        max_seq_len=seq_len,
        num_tags=0,
        grid_size=grid_size,
        struct_cond=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Conditioned AR Transformer parameters: {n_params:,}")

    # Resume from v2 AR checkpoint if available (transfer learning)
    start_step = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        # Load compatible weights (skip struct_* layers that don't exist in v2)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in ckpt['model_state_dict'].items()
                          if k in model_dict and v.shape == model_dict[k].shape}
        n_loaded = len(pretrained_dict)
        n_total = len(model_dict)
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"Loaded {n_loaded}/{n_total} weight tensors from checkpoint")
        if args.resume_step:
            start_step = args.resume_step
            print(f"Resuming from step {start_step}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # LR schedule
    total_steps = args.steps
    def lr_schedule(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Fast-forward scheduler if resuming
    if start_step > 0:
        for _ in range(start_step):
            scheduler.step()

    scaler = torch.amp.GradScaler('cuda')

    # Checkpoint dir
    ckpt_dir = PROJECT_ROOT / "checkpoints" / (args.ckpt_dir or "ar_cond")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    model.train()
    data_iter = iter(loader)
    step = start_step
    log_interval = 50
    save_interval = args.save_every

    start_time = time.time()
    running_loss = 0
    running_acc = 0
    running_top5_acc = 0

    print(f"\nStarting conditioned AR training...")
    print(f"Batch size: {args.batch_size}, Steps: {args.steps}")
    print(f"Sequence length: {seq_len} ({grid_size}x{grid_size}x{grid_size})")
    print(f"Structural conditioning: 6 features, CFG dropout {args.uncond_prob:.0%}")
    print(f"Checkpoints saved to: {ckpt_dir}")
    print("-" * 80)

    while step < args.steps:
        try:
            codes, features = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            codes, features = next(data_iter)

        codes = codes.to(device)        # (B, 512)
        features = features.to(device)  # (B, 6)

        # Teacher-forced AR
        bos = torch.full((codes.shape[0], 1), model.bos_token_id,
                         dtype=torch.long, device=device)
        input_seq = torch.cat([bos, codes[:, :-1]], dim=1)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(input_seq, struct_features=features)  # (B, 512, num_codes)
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
            top5 = logits.topk(5, dim=-1).indices
            top5_match = (top5 == codes.unsqueeze(-1)).any(dim=-1).float().mean().item()

        running_loss += loss.item()
        running_acc += acc
        running_top5_acc += top5_match
        step += 1

        if step % log_interval == 0:
            elapsed = time.time() - start_time
            steps_done = step - start_step
            steps_per_sec = steps_done / elapsed
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
            ckpt_path = ckpt_dir / f"ar_cond_step{step}.pt"
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'args': vars(args),
                'num_codes': num_codes,
                'struct_cond': True,
                'struct_feature_vocabs': ARTransformer3D.STRUCT_FEATURE_VOCABS,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    total_time = time.time() - start_time
    print(f"\nConditioned AR training complete in {total_time/60:.1f} minutes")
    print(f"Final checkpoint: {ckpt_dir / f'ar_cond_step{args.steps}.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--steps', type=int, default=60000)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--dim', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=12)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--uncond_prob', type=float, default=0.1)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--resume_step', type=int, default=0)
    parser.add_argument('--latent_path', type=str, default=None,
                        help="Override latent codes path (for augmentation experiments)")
    parser.add_argument('--features_path', type=str, default=None,
                        help="Override features path (for augmentation experiments)")
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help="Override checkpoint directory name")
    train(parser.parse_args())
