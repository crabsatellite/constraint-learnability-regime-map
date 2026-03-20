"""
Generate Minecraft builds using trained VQ-VAE + AR Transformer.

Generates latent codes with AR model, decodes with VQ-VAE to voxel grids,
and saves as .npz files + renders 3D visualizations.
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.vqvae import VQVAE3D
from models.ar_transformer import ARTransformer3D


def load_models(vqvae_ckpt, ar_ckpt, device='cuda'):
    """Load trained VQ-VAE and AR models."""

    # Load VQ-VAE
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

    # Load AR — infer grid_size from positional embedding shape
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
    ).to(device)
    ar.load_state_dict(ar_data['model_state_dict'])
    ar.eval()
    print(f"AR model: grid_size={grid_size}, seq_len={seq_len}")

    return vqvae, ar, grid_size


def generate_builds(vqvae, ar, grid_size=4, n_samples=50, temperature=0.9,
                    top_k=100, device='cuda'):
    """Generate builds: AR -> latent codes -> VQ-VAE decode -> voxels."""
    print(f"Generating {n_samples} samples (temp={temperature}, top_k={top_k})...")

    all_voxels = []
    batch_size = min(8, n_samples)

    for start in range(0, n_samples, batch_size):
        bs = min(batch_size, n_samples - start)
        print(f"  Generating batch {start//batch_size + 1}...")

        # Generate latent codes
        codes = ar.generate_batch(
            batch_size=bs,
            temperature=temperature,
            top_k=top_k,
            device=device,
        )  # (bs, grid_size^3)

        # Reshape to grid
        code_grid = codes.reshape(bs, grid_size, grid_size, grid_size)

        # Decode through VQ-VAE
        logits = vqvae.decode_from_indices(code_grid)  # (bs, vocab, 32, 32, 32)
        voxels = logits.argmax(dim=1).cpu().numpy()  # (bs, 32, 32, 32)

        all_voxels.append(voxels)

    return np.concatenate(all_voxels, axis=0)


def render_voxels_matplotlib(voxels, save_path, title=""):
    """Simple 3D voxel rendering using matplotlib."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Non-air voxels
    filled = voxels > 0

    if not filled.any():
        plt.close()
        return

    # Color by block type (simple colormap)
    colors = np.zeros((*voxels.shape, 4))
    non_air = voxels[filled]
    unique_blocks = np.unique(non_air)

    # Generate distinct colors
    cmap = plt.cm.Set3(np.linspace(0, 1, max(len(unique_blocks), 1)))
    block_to_color = {b: cmap[i % len(cmap)] for i, b in enumerate(unique_blocks)}

    xs, ys, zs = np.where(filled)
    for x, y, z in zip(xs, ys, zs):
        colors[x, y, z] = block_to_color[voxels[x, y, z]]

    ax.voxels(filled, facecolors=colors, edgecolors=colors * 0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # Better viewing angle
    ax.view_init(elev=25, azim=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_samples(voxels, output_dir):
    """Save generated voxels as .npz and render images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    render_dir = output_dir / "renders"
    render_dir.mkdir(exist_ok=True)

    for i, v in enumerate(voxels):
        # Save voxel data
        np.savez_compressed(output_dir / f"sample_{i:04d}.npz", voxels=v)

        # Count non-air blocks
        n_blocks = (v > 0).sum()
        n_unique = len(np.unique(v[v > 0])) if n_blocks > 0 else 0

        # Render
        try:
            render_voxels_matplotlib(
                v, render_dir / f"sample_{i:04d}.png",
                title=f"Sample {i} ({n_blocks} blocks, {n_unique} types)"
            )
        except Exception as e:
            print(f"  Warning: render failed for sample {i}: {e}")

    print(f"Saved {len(voxels)} samples to {output_dir}")
    print(f"Renders saved to {render_dir}")

    # Summary stats
    block_counts = [(v > 0).sum() for v in voxels]
    print(f"\nGeneration stats:")
    print(f"  Avg blocks per build: {np.mean(block_counts):.0f}")
    print(f"  Min: {min(block_counts)}, Max: {max(block_counts)}")
    print(f"  Empty builds: {sum(1 for c in block_counts if c == 0)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vqvae_ckpt', type=str, required=True)
    parser.add_argument('--ar_ckpt', type=str, required=True)
    parser.add_argument('--n_samples', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=0.9)
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--output_dir', type=str,
                        default=str(PROJECT_ROOT / "outputs" / "generated"))
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vqvae, ar, grid_size = load_models(args.vqvae_ckpt, args.ar_ckpt, device)
    voxels = generate_builds(vqvae, ar, grid_size, args.n_samples,
                             args.temperature, args.top_k, device)
    save_samples(voxels, args.output_dir)


if __name__ == "__main__":
    main()
