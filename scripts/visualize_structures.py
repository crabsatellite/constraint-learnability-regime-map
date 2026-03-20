"""
Visualize generated 3D structures for paper figures.

Generates samples under key conditions and renders isometric voxel views.
Outputs PNG figures suitable for paper inclusion.

Requires: model checkpoints (vqvae + ar_cond)
"""

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.test_regime_map import load_models, make_features, measure_all_properties


# Block type color palette (Minecraft-inspired)
BLOCK_COLORS = {
    0: None,          # air = transparent
    1: '#8B8B8B',     # stone
    2: '#7B9A3A',     # grass
    3: '#866043',     # dirt
    4: '#AAAAAA',     # cobblestone
    5: '#BC9862',     # oak planks
    6: '#2B5F2B',     # leaves
    7: '#C8C8C8',     # glass (light)
    8: '#3366CC',     # water
    9: '#CC6633',     # lava
}
DEFAULT_COLOR = '#D4A76A'  # sand/default


def render_voxels(voxels, ax, title=None, elev=25, azim=45):
    """Render a 32x32x32 voxel grid as 3D filled cubes."""
    filled = voxels > 0
    if not filled.any():
        ax.set_title(title or "(empty)")
        return

    # Find bounding box to crop
    coords = np.argwhere(filled)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    cropped = voxels[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]

    # Build color array
    sx, sy, sz = cropped.shape
    colors = np.empty(cropped.shape, dtype=object)
    for x in range(sx):
        for y in range(sy):
            for z in range(sz):
                bid = int(cropped[x, y, z])
                colors[x, y, z] = BLOCK_COLORS.get(bid, DEFAULT_COLOR) if bid > 0 else ''

    filled_crop = cropped > 0
    facecolors = np.where(filled_crop, colors, '')

    ax.voxels(filled_crop, facecolors=facecolors, edgecolor='#00000020', linewidth=0.2)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(0, sx)
    ax.set_ylim(0, sy)
    ax.set_zlim(0, sz)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=9, pad=-5)


def generate_samples(ar, vqvae, grid_size, spec, cfg_scale=2.0, seed=42,
                     n_samples=4, device='cuda'):
    """Generate n_samples voxel grids under a condition."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    features = make_features(spec, device, n_samples)
    codes = ar.generate_batch(
        batch_size=n_samples, struct_features=features,
        temperature=0.9, top_k=100, device=device, cfg_scale=cfg_scale,
    )
    code_grid = codes.reshape(n_samples, grid_size, grid_size, grid_size)
    logits = vqvae.decode_from_indices(code_grid)
    voxels_batch = logits.argmax(dim=1).cpu().numpy()
    return voxels_batch


def make_comparison_figure(ar, vqvae, grid_size, device='cuda'):
    """Main figure: unconditioned vs key conditions, 4 samples each."""
    conditions = {
        'Unconditioned': {},
        'Height=Tower': {'height': 3},
        'Size=Large': {'size': 3},
        'Symmetry=1': {'symmetry': 1},
        'Enclosure=1': {'enclosure': 1},
    }

    n_conds = len(conditions)
    n_samples = 4
    fig = plt.figure(figsize=(n_samples * 3, n_conds * 3))

    for row, (cond_name, spec) in enumerate(conditions.items()):
        voxels_batch = generate_samples(ar, vqvae, grid_size, spec, n_samples=n_samples, device=device)

        for col in range(n_samples):
            ax = fig.add_subplot(n_conds, n_samples, row * n_samples + col + 1, projection='3d')
            title = cond_name if col == 0 else None
            render_voxels(voxels_batch[col], ax, title=title)

            # Annotate key property for first sample
            if col == 0:
                props = measure_all_properties(voxels_batch[col])
                key_props = {
                    'Unconditioned': f"h={props['height']}",
                    'Height=Tower': f"h={props['height']}",
                    'Size=Large': f"n={props['n_blocks']}",
                    'Symmetry=1': f"sym={props['symmetry_iou']:.2f}",
                    'Enclosure=1': f"enc={props['enclosed_ratio']:.3f}",
                }
                ax.text2D(0.02, 0.02, key_props.get(cond_name, ''),
                          transform=ax.transAxes, fontsize=7, color='gray')

    fig.suptitle('Structural Constraint Controllability: Unconditioned vs Conditioned Samples',
                 fontsize=12, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def make_cfg_comparison_figure(ar, vqvae, grid_size, device='cuda'):
    """CFG sensitivity figure: same condition at cfg=0, 2, 4."""
    cfg_scales = [0.0, 2.0, 4.0]
    conditions = {
        'Enclosure=1': {'enclosure': 1},
        'Height=Tower': {'height': 3},
    }

    n_cfgs = len(cfg_scales)
    n_conds = len(conditions)
    fig = plt.figure(figsize=(n_cfgs * 3.5, n_conds * 3.5))

    for row, (cond_name, spec) in enumerate(conditions.items()):
        for col, cfg in enumerate(cfg_scales):
            voxels_batch = generate_samples(ar, vqvae, grid_size, spec,
                                            cfg_scale=cfg, n_samples=1, device=device)
            ax = fig.add_subplot(n_conds, n_cfgs, row * n_cfgs + col + 1, projection='3d')
            title = f"{cond_name}\nCFG={cfg}"
            render_voxels(voxels_batch[0], ax, title=title)

            props = measure_all_properties(voxels_batch[0])
            if 'Enclosure' in cond_name:
                ax.text2D(0.02, 0.02, f"enc={props['enclosed_ratio']:.3f}",
                          transform=ax.transAxes, fontsize=7, color='gray')

    fig.suptitle('CFG Sensitivity: Guidance Scale Effect on Structure',
                 fontsize=12, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vqvae_ckpt = str(PROJECT_ROOT / "checkpoints" / "vqvae" / "vqvae_step100000.pt")
    ar_ckpt = str(PROJECT_ROOT / "checkpoints" / "ar_cond" / "ar_cond_step80000.pt")

    print(f"Loading models (device={device})...")
    vqvae, ar, grid_size = load_models(vqvae_ckpt, ar_ckpt, device)

    fig_dir = PROJECT_ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)

    print("Generating comparison figure...")
    fig1 = make_comparison_figure(ar, vqvae, grid_size, device)
    fig1.savefig(fig_dir / "condition_comparison.png", dpi=200, bbox_inches='tight')
    fig1.savefig(fig_dir / "condition_comparison.pdf", bbox_inches='tight')
    print(f"  Saved: {fig_dir / 'condition_comparison.png'}")

    print("Generating CFG sensitivity figure...")
    fig2 = make_cfg_comparison_figure(ar, vqvae, grid_size, device)
    fig2.savefig(fig_dir / "cfg_sensitivity.png", dpi=200, bbox_inches='tight')
    fig2.savefig(fig_dir / "cfg_sensitivity.pdf", bbox_inches='tight')
    print(f"  Saved: {fig_dir / 'cfg_sensitivity.png'}")

    plt.close('all')
    print("\nDone. Figures saved to:", fig_dir)


if __name__ == "__main__":
    main()
