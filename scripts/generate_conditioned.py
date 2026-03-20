"""
Generate Minecraft builds with structural constraint conditioning.

Demonstrates controllable generation:
  - Specify height, size, footprint, symmetry, enclosure, complexity
  - Classifier-free guidance for stronger conditioning adherence
  - Generates constrained samples + unconditional baselines for comparison
"""

import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.vqvae import VQVAE3D
from models.ar_transformer import ARTransformer3D


# Human-readable labels for structural features
FEATURE_LABELS = {
    'height': {0: 'flat (1-8)', 1: 'medium (9-16)', 2: 'tall (17-24)', 3: 'tower (25-32)'},
    'size': {0: 'tiny (1-100)', 1: 'small (101-500)', 2: 'medium (501-2K)', 3: 'large (2K+)'},
    'footprint': {0: 'compact', 1: 'medium', 2: 'sprawling'},
    'symmetry': {0: 'asymmetric', 1: 'symmetric'},
    'enclosure': {0: 'open', 1: 'enclosed'},
    'complexity': {0: 'blocky', 1: 'moderate', 2: 'detailed'},
}


def load_models(vqvae_ckpt, ar_ckpt, device='cuda'):
    """Load VQ-VAE and conditioned AR models."""
    # VQ-VAE
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

    # Conditioned AR
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
    print(f"Conditioned AR model: grid_size={grid_size}, seq_len={seq_len}")

    return vqvae, ar, grid_size


def make_features_tensor(specs, batch_size, device):
    """Create structural features tensor from spec dict.

    Args:
        specs: dict of {feature_name: value} or None for unconditional
        batch_size: number of samples
        device: torch device
    Returns:
        features: (batch_size, 6) tensor
    """
    vocabs = ARTransformer3D.STRUCT_FEATURE_VOCABS
    names = ARTransformer3D.STRUCT_FEATURE_NAMES
    uncond_ids = [v for v in vocabs]

    features = []
    for i, name in enumerate(names):
        if specs and name in specs:
            features.append(specs[name])
        else:
            features.append(uncond_ids[i])  # unconditional

    return torch.tensor([features] * batch_size, dtype=torch.long, device=device)


def generate_with_constraints(vqvae, ar, grid_size, specs, n_samples=8,
                               temperature=0.9, top_k=100, cfg_scale=2.0,
                               device='cuda'):
    """Generate builds matching structural constraints."""
    features = make_features_tensor(specs, n_samples, device)

    all_voxels = []
    batch_size = min(8, n_samples)

    for start in range(0, n_samples, batch_size):
        bs = min(batch_size, n_samples - start)
        feat_batch = features[start:start+bs]

        codes = ar.generate_batch(
            batch_size=bs,
            struct_features=feat_batch,
            temperature=temperature,
            top_k=top_k,
            device=device,
            cfg_scale=cfg_scale,
        )
        code_grid = codes.reshape(bs, grid_size, grid_size, grid_size)
        logits = vqvae.decode_from_indices(code_grid)
        voxels = logits.argmax(dim=1).cpu().numpy()
        all_voxels.append(voxels)

    return np.concatenate(all_voxels, axis=0)


def render_voxels_matplotlib(voxels, save_path, title=""):
    """3D voxel rendering."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    filled = voxels > 0
    if not filled.any():
        plt.close()
        return

    colors = np.zeros((*voxels.shape, 4))
    non_air = voxels[filled]
    unique_blocks = np.unique(non_air)
    cmap = plt.cm.Set3(np.linspace(0, 1, max(len(unique_blocks), 1)))
    block_to_color = {b: cmap[i % len(cmap)] for i, b in enumerate(unique_blocks)}

    xs, ys, zs = np.where(filled)
    for x, y, z in zip(xs, ys, zs):
        colors[x, y, z] = block_to_color[voxels[x, y, z]]

    ax.voxels(filled, facecolors=colors, edgecolors=colors * 0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=25, azim=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vqvae_ckpt', type=str, required=True)
    parser.add_argument('--ar_ckpt', type=str, required=True)
    parser.add_argument('--n_samples', type=int, default=8,
                        help="Samples per constraint configuration")
    parser.add_argument('--temperature', type=float, default=0.9)
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--cfg_scale', type=float, default=2.0,
                        help="Classifier-free guidance scale (0=off)")
    parser.add_argument('--output_dir', type=str,
                        default=str(PROJECT_ROOT / "outputs" / "conditioned"))
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vqvae, ar, grid_size = load_models(args.vqvae_ckpt, args.ar_ckpt, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define constraint experiments for the paper
    experiments = {
        # 1. Unconditional baseline
        'uncond': {},

        # 2. Height control
        'height_flat': {'height': 0},
        'height_tall': {'height': 2},
        'height_tower': {'height': 3},

        # 3. Size control
        'size_tiny': {'size': 0},
        'size_large': {'size': 3},

        # 4. Symmetry control
        'symmetric': {'symmetry': 1},
        'asymmetric': {'symmetry': 0},

        # 5. Enclosure control
        'enclosed': {'enclosure': 1},
        'open': {'enclosure': 0},

        # 6. Complexity control
        'blocky': {'complexity': 0},
        'detailed': {'complexity': 2},

        # 7. Combined constraints (the real test)
        'tall_symmetric_enclosed': {'height': 2, 'symmetry': 1, 'enclosure': 1},
        'flat_sprawling_detailed': {'height': 0, 'footprint': 2, 'complexity': 2},
        'tower_compact_blocky': {'height': 3, 'footprint': 0, 'complexity': 0},
        'large_symmetric_detailed': {'size': 3, 'symmetry': 1, 'complexity': 2},
    }

    results = {}

    for exp_name, specs in experiments.items():
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_name}")
        if specs:
            for k, v in specs.items():
                print(f"  {k}: {FEATURE_LABELS[k][v]}")
        else:
            print("  Unconditional")
        print(f"{'='*60}")

        voxels = generate_with_constraints(
            vqvae, ar, grid_size, specs,
            n_samples=args.n_samples,
            temperature=args.temperature,
            top_k=args.top_k,
            cfg_scale=args.cfg_scale if specs else 0.0,
            device=device,
        )

        # Save and analyze
        exp_dir = output_dir / exp_name
        exp_dir.mkdir(exist_ok=True)
        render_dir = exp_dir / "renders"
        render_dir.mkdir(exist_ok=True)

        block_counts = []
        heights = []
        for i, v in enumerate(voxels):
            np.savez_compressed(exp_dir / f"sample_{i:04d}.npz", voxels=v)
            n_blocks = int((v > 0).sum())
            n_types = len(np.unique(v[v > 0])) if n_blocks > 0 else 0
            block_counts.append(n_blocks)

            # Measure actual height
            if n_blocks > 0:
                ys = np.where(v > 0)[1]
                h = int(ys.max() - ys.min() + 1)
            else:
                h = 0
            heights.append(h)

            # Build constraint label for title
            label_parts = [f"{FEATURE_LABELS[k][v_val]}" for k, v_val in specs.items()] if specs else ['uncond']
            try:
                render_voxels_matplotlib(
                    v, render_dir / f"sample_{i:04d}.png",
                    title=f"{exp_name} #{i} ({n_blocks}blk, h={h}, {n_types}types)"
                )
            except Exception as e:
                print(f"  Render failed for {i}: {e}")

        stats = {
            'experiment': exp_name,
            'constraints': specs,
            'n_samples': len(voxels),
            'avg_blocks': float(np.mean(block_counts)),
            'avg_height': float(np.mean(heights)),
            'std_height': float(np.std(heights)),
            'min_blocks': int(min(block_counts)),
            'max_blocks': int(max(block_counts)),
        }
        results[exp_name] = stats
        print(f"  Avg blocks: {stats['avg_blocks']:.0f}, Avg height: {stats['avg_height']:.1f}")

    # Save results summary
    with open(output_dir / "conditioned_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"{'Experiment':<30} {'Avg Blk':>8} {'Avg H':>6} {'Std H':>6} {'Min':>6} {'Max':>6}")
    print(f"{'='*80}")
    for name, s in results.items():
        print(f"{name:<30} {s['avg_blocks']:>8.0f} {s['avg_height']:>6.1f} "
              f"{s['std_height']:>6.1f} {s['min_blocks']:>6d} {s['max_blocks']:>6d}")
    print(f"{'='*80}")

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
