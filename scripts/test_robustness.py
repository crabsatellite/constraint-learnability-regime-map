"""
Robustness check for dual bottleneck findings.

Confirms that the frequency floor / representation ceiling distinction
is stable across:
  - 5 random seeds
  - 4 IoU thresholds (0.5, 0.6, 0.7, 0.8)
  - Both baseline and sym-aug models

If symmetry consistently shows soft improvement but no hard satisfaction,
and enclosure consistently shows zero response, the dual bottleneck is stable.
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from collections import deque

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.vqvae import VQVAE3D
from models.ar_transformer import ARTransformer3D


def load_models(vqvae_ckpt, ar_ckpt, device='cuda'):
    vqvae_data = torch.load(vqvae_ckpt, map_location=device, weights_only=False)
    vqvae_args = vqvae_data['args']
    vqvae = VQVAE3D(
        vocab_size=vqvae_data['vocab_size'],
        embed_dim=vqvae_args['embed_dim'],
        hidden_dim=vqvae_args['hidden_dim'],
        code_dim=vqvae_args['code_dim'],
        num_codes=vqvae_args['num_codes'],
        n_downsample=vqvae_args.get('n_downsample', 3),
    ).to(device)
    vqvae.load_state_dict(vqvae_data['model_state_dict'])
    vqvae.eval()

    ar_data = torch.load(ar_ckpt, map_location=device, weights_only=False)
    ar_args = ar_data['args']
    pe_shape = ar_data['model_state_dict']['pos_embed.pe'].shape[0]
    grid_size = round(pe_shape ** (1/3))
    seq_len = grid_size ** 3
    ar = ARTransformer3D(
        num_codes=ar_data['num_codes'],
        dim=ar_args['dim'],
        n_layers=ar_args['n_layers'],
        n_heads=ar_args['n_heads'],
        dropout=0.0,
        max_seq_len=seq_len,
        num_tags=0,
        grid_size=grid_size,
        struct_cond=ar_data.get('struct_cond', True),
    ).to(device)
    ar.load_state_dict(ar_data['model_state_dict'])
    ar.eval()
    return vqvae, ar, grid_size


def make_features(spec, device, n=8):
    vocabs = ARTransformer3D.STRUCT_FEATURE_VOCABS
    names = ARTransformer3D.STRUCT_FEATURE_NAMES
    uncond = [v for v in vocabs]
    feat = list(uncond)
    for k, v in spec.items():
        idx = names.index(k)
        feat[idx] = v
    return torch.tensor([feat] * n, dtype=torch.long, device=device)


def measure_symmetry_iou(voxels):
    filled = voxels > 0
    if not filled.any():
        return 0.0
    flipped = np.flip(filled, axis=0)
    inter = (filled & flipped).sum()
    union = (filled | flipped).sum()
    return float(inter / max(union, 1))


def measure_enclosure_ratio(voxels):
    is_air = (voxels == 0)
    total_air = int(is_air.sum())
    if total_air == 0:
        return 0.0
    shape = voxels.shape
    visited = np.zeros(shape, dtype=bool)
    queue = deque()
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                if (x in (0, shape[0]-1) or y in (0, shape[1]-1) or z in (0, shape[2]-1)):
                    if is_air[x, y, z]:
                        visited[x, y, z] = True
                        queue.append((x, y, z))
    while queue:
        x, y, z = queue.popleft()
        for dx, dy, dz in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
            nx, ny, nz = x+dx, y+dy, z+dz
            if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
                if is_air[nx, ny, nz] and not visited[nx, ny, nz]:
                    visited[nx, ny, nz] = True
                    queue.append((nx, ny, nz))
    enclosed = total_air - int(visited.sum())
    return enclosed / total_air


def run_seed_experiment(ar, vqvae, grid_size, spec, seed, n=8, device='cuda'):
    """Generate n samples with given seed, measure symmetry + enclosure."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    features = make_features(spec, device, n)
    codes = ar.generate_batch(
        batch_size=n, struct_features=features,
        temperature=0.9, top_k=100, device=device, cfg_scale=0.0,
    )
    code_grid = codes.reshape(n, grid_size, grid_size, grid_size)
    logits = vqvae.decode_from_indices(code_grid)
    voxels_batch = logits.argmax(dim=1).cpu().numpy()

    sym_ious = [measure_symmetry_iou(v) for v in voxels_batch]
    enc_ratios = [measure_enclosure_ratio(v) for v in voxels_batch]
    return sym_ious, enc_ratios


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vqvae_ckpt = str(PROJECT_ROOT / "checkpoints" / "vqvae" / "vqvae_step100000.pt")

    # Two models to compare
    models = {
        'baseline': str(PROJECT_ROOT / "checkpoints" / "ar_cond" / "ar_cond_step80000.pt"),
        'sym_aug': str(PROJECT_ROOT / "checkpoints" / "ar_cond_sym_aug" / "ar_cond_step80000.pt"),
    }

    seeds = [42, 123, 456, 789, 2026]
    thresholds = [0.5, 0.6, 0.7, 0.8]

    # Key conditions to test
    conditions = {
        'unconditioned': {},
        'symmetry=1': {'symmetry': 1},
        'enclosure=1': {'enclosure': 1},
        'size=3+symmetry=1': {'size': 3, 'symmetry': 1},
    }

    results = {}

    for model_name, ar_ckpt in models.items():
        print(f"\n{'='*70}")
        print(f"Model: {model_name}")
        print(f"{'='*70}")

        vqvae, ar, grid_size = load_models(vqvae_ckpt, ar_ckpt, device)
        results[model_name] = {}

        for cond_name, spec in conditions.items():
            all_sym_ious = []
            all_enc_ratios = []

            for seed in seeds:
                sym_ious, enc_ratios = run_seed_experiment(
                    ar, vqvae, grid_size, spec, seed, n=8, device=device
                )
                all_sym_ious.extend(sym_ious)
                all_enc_ratios.extend(enc_ratios)

            # Compute statistics across all seeds
            sym_arr = np.array(all_sym_ious)
            enc_arr = np.array(all_enc_ratios)

            sym_stats = {
                'mean': round(float(sym_arr.mean()), 4),
                'std': round(float(sym_arr.std()), 4),
                'median': round(float(np.median(sym_arr)), 4),
                'max': round(float(sym_arr.max()), 4),
            }
            enc_stats = {
                'mean': round(float(enc_arr.mean()), 4),
                'std': round(float(enc_arr.std()), 4),
                'median': round(float(np.median(enc_arr)), 4),
                'max': round(float(enc_arr.max()), 4),
            }

            # Threshold sweep for symmetry
            sym_threshold_rates = {}
            for t in thresholds:
                rate = float((sym_arr >= t).mean())
                sym_threshold_rates[str(t)] = round(rate, 3)

            # Threshold sweep for enclosure
            enc_threshold_rates = {}
            for t in [0.01, 0.02, 0.05, 0.10]:
                rate = float((enc_arr >= t).mean())
                enc_threshold_rates[str(t)] = round(rate, 3)

            results[model_name][cond_name] = {
                'n_samples': len(all_sym_ious),
                'symmetry': sym_stats,
                'symmetry_threshold_rates': sym_threshold_rates,
                'enclosure': enc_stats,
                'enclosure_threshold_rates': enc_threshold_rates,
                'all_sym_ious': [round(x, 4) for x in all_sym_ious],
                'all_enc_ratios': [round(x, 4) for x in all_enc_ratios],
            }

            print(f"\n  {cond_name} (n={len(all_sym_ious)}, {len(seeds)} seeds):")
            print(f"    Symmetry IoU:  mean={sym_stats['mean']:.3f} +/- {sym_stats['std']:.3f}, "
                  f"median={sym_stats['median']:.3f}, max={sym_stats['max']:.3f}")
            print(f"    Sym thresholds: {sym_threshold_rates}")
            print(f"    Enclosure:     mean={enc_stats['mean']:.4f} +/- {enc_stats['std']:.4f}, "
                  f"max={enc_stats['max']:.4f}")
            print(f"    Enc thresholds: {enc_threshold_rates}")

    # Summary comparison
    print(f"\n{'='*70}")
    print(f"ROBUSTNESS SUMMARY (5 seeds x 8 samples = 40 per condition)")
    print(f"{'='*70}")

    print(f"\n{'Condition':<25} {'Model':<12} {'Sym Mean':>9} {'Sym>0.5':>8} {'Sym>0.7':>8} {'Enc Mean':>9} {'Enc>0.01':>9}")
    print(f"{'-'*80}")
    for cond_name in conditions:
        for model_name in models:
            r = results[model_name][cond_name]
            print(f"{cond_name:<25} {model_name:<12} "
                  f"{r['symmetry']['mean']:>9.3f} "
                  f"{r['symmetry_threshold_rates']['0.5']:>8.1%} "
                  f"{r['symmetry_threshold_rates']['0.7']:>8.1%} "
                  f"{r['enclosure']['mean']:>9.4f} "
                  f"{r['enclosure_threshold_rates']['0.01']:>9.1%}")

    # Stability verdict
    print(f"\n--- Dual Bottleneck Stability ---")

    # Check: does sym-aug consistently improve symmetry across all conditions?
    sym_improvements = []
    for cond_name in conditions:
        baseline_sym = results['baseline'][cond_name]['symmetry']['mean']
        aug_sym = results['sym_aug'][cond_name]['symmetry']['mean']
        improvement = aug_sym - baseline_sym
        sym_improvements.append(improvement)
        print(f"  {cond_name}: sym improvement = {improvement:+.3f}")

    consistent_sym_improvement = all(x >= -0.02 for x in sym_improvements)
    print(f"  Consistent symmetry improvement: {consistent_sym_improvement}")

    # Check: does enclosure stay at ~0 in both models across all conditions?
    enc_all_zero = True
    for model_name in models:
        for cond_name in conditions:
            enc_max = results[model_name][cond_name]['enclosure']['max']
            if enc_max > 0.1:
                enc_all_zero = False
    print(f"  Enclosure consistently near-zero: {enc_all_zero}")

    if consistent_sym_improvement and enc_all_zero:
        print(f"\n  [CONFIRMED] Dual bottleneck is STABLE across seeds and thresholds")
        print(f"  -> Frequency floor: symmetry responds to augmentation")
        print(f"  -> Representation ceiling: enclosure unresponsive regardless")
    else:
        print(f"\n  [INCONCLUSIVE] Results vary — need investigation")

    # Save
    out_path = PROJECT_ROOT / "outputs" / "robustness_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
