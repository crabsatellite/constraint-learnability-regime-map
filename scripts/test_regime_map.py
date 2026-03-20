"""
Constraint Learnability Regime Map — expanded taxonomy (12 properties).

Original 7: height, n_blocks, symmetry_iou, enclosed_ratio, elongation, floor_count, hollowness
New 5: bbox_volume, surface_ratio, connected_components, vertical_aspect, z_symmetry_iou

The expansion increases regime map resolution from n=7 to n=12, enabling
meaningful leave-one-out validation of the learnability predictor.

No retraining needed — pure measurement on existing models.
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
        num_codes=ar_data['num_codes'], dim=ar_args['dim'],
        n_layers=ar_args['n_layers'], n_heads=ar_args['n_heads'],
        dropout=0.0, max_seq_len=seq_len, num_tags=0,
        grid_size=grid_size, struct_cond=ar_data.get('struct_cond', True),
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


# ============================================================
# Property measurements
# ============================================================

def measure_all_properties(voxels):
    """Compute all 12 structural properties for a single 32x32x32 voxel grid."""
    filled = voxels > 0
    n_blocks = int(filled.sum())
    all_props = [
        'height', 'n_blocks', 'symmetry_iou', 'enclosed_ratio',
        'elongation', 'floor_count', 'hollowness',
        'bbox_volume', 'surface_ratio', 'connected_components',
        'vertical_aspect', 'z_symmetry_iou',
        'layer_consistency', 'footprint_convexity',
    ]
    if n_blocks == 0:
        return {k: 0.0 for k in all_props}

    # --- Existing properties ---
    # Height
    ys = np.where(filled.any(axis=(0, 2)))[0]
    height = int(ys[-1] - ys[0] + 1) if len(ys) > 0 else 0

    # Symmetry IoU
    flipped = np.flip(filled, axis=0)
    sym_inter = (filled & flipped).sum()
    sym_union = (filled | flipped).sum()
    symmetry_iou = float(sym_inter / max(sym_union, 1))

    # Enclosure
    is_air = ~filled
    total_air = int(is_air.sum())
    if total_air > 0:
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
        enclosed_ratio = enclosed / total_air
    else:
        enclosed_ratio = 0.0

    # --- New intermediate properties ---

    # Elongation: footprint aspect ratio (max/min of XZ bounding box)
    xz = filled.any(axis=1)  # (X, Z)
    xs = np.where(xz.any(axis=1))[0]
    zs = np.where(xz.any(axis=0))[0]
    if len(xs) > 0 and len(zs) > 0:
        x_span = xs[-1] - xs[0] + 1
        z_span = zs[-1] - zs[0] + 1
        elongation = max(x_span, z_span) / max(min(x_span, z_span), 1)
    else:
        elongation = 1.0

    # Floor count: number of distinct Y layers with significant block presence
    # A "floor" = Y layer where >10% of the footprint XZ area is filled
    if len(xs) > 0 and len(zs) > 0:
        footprint_area = (xs[-1] - xs[0] + 1) * (zs[-1] - zs[0] + 1)
        y_min, y_max = int(ys[0]), int(ys[-1])
        floor_count = 0
        for y in range(y_min, y_max + 1):
            layer = filled[:, y, :]
            fill_ratio = layer.sum() / max(footprint_area, 1)
            if fill_ratio > 0.1:
                floor_count += 1
    else:
        floor_count = 0

    # Hollowness: interior air within bounding box / total bounding box volume
    # Weaker than enclosure — doesn't require topological closure,
    # just measures how much of the bounding box is air
    if len(xs) > 0 and len(ys) > 0 and len(zs) > 0:
        bbox = filled[xs[0]:xs[-1]+1, ys[0]:ys[-1]+1, zs[0]:zs[-1]+1]
        bbox_vol = bbox.size
        bbox_solid = int(bbox.sum())
        hollowness = 1.0 - (bbox_solid / max(bbox_vol, 1))
    else:
        hollowness = 0.0

    # --- New properties (expansion to 12) ---

    # Bounding box volume: total spatial extent (LOCAL)
    bbox_volume = int(x_span * height * z_span) if (len(xs) > 0 and len(zs) > 0) else 0

    # Surface ratio: fraction of blocks with at least one air neighbor (SEMI-LOCAL)
    # Measures how "thin-walled" vs "solid" the structure is
    surface_count = 0
    if n_blocks > 0:
        shape = voxels.shape
        padded_filled = np.pad(filled, 1, mode='constant', constant_values=False)
        for dx, dy, dz in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
            shifted = padded_filled[1+dx:shape[0]+1+dx, 1+dy:shape[1]+1+dy, 1+dz:shape[2]+1+dz]
            surface_count += (filled & ~shifted).sum()
        surface_blocks = (surface_count > 0).item() if isinstance(surface_count, np.ndarray) else surface_count
        # Actually compute it properly: count blocks that have at least 1 air neighbor
        has_air_neighbor = np.zeros_like(filled)
        for dx, dy, dz in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
            shifted = padded_filled[1+dx:shape[0]+1+dx, 1+dy:shape[1]+1+dy, 1+dz:shape[2]+1+dz]
            has_air_neighbor |= (filled & ~shifted)
        surface_blocks_count = int(has_air_neighbor.sum())
        surface_ratio = surface_blocks_count / max(n_blocks, 1)
    else:
        surface_ratio = 0.0

    # Connected components: count of 6-connected components (GLOBAL/TOPOLOGICAL)
    if n_blocks > 0:
        component_label = np.zeros_like(filled, dtype=int)
        comp_id = 0
        for x_c in range(shape[0]):
            for y_c in range(shape[1]):
                for z_c in range(shape[2]):
                    if filled[x_c, y_c, z_c] and component_label[x_c, y_c, z_c] == 0:
                        comp_id += 1
                        bfs_queue = deque([(x_c, y_c, z_c)])
                        component_label[x_c, y_c, z_c] = comp_id
                        while bfs_queue:
                            cx, cy, cz = bfs_queue.popleft()
                            for dx, dy, dz in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
                                nx, ny, nz = cx+dx, cy+dy, cz+dz
                                if (0 <= nx < shape[0] and 0 <= ny < shape[1] and
                                    0 <= nz < shape[2] and filled[nx, ny, nz] and
                                    component_label[nx, ny, nz] == 0):
                                    component_label[nx, ny, nz] = comp_id
                                    bfs_queue.append((nx, ny, nz))
        connected_components = comp_id
    else:
        connected_components = 0

    # Vertical aspect: height / max(x_span, z_span) (SEMI-LOCAL)
    # Captures how "tower-like" vs "flat" independent of horizontal elongation
    if len(xs) > 0 and len(zs) > 0:
        vertical_aspect = height / max(max(x_span, z_span), 1)
    else:
        vertical_aspect = 0.0

    # Z-axis symmetry IoU: same as symmetry_iou but along Z axis (GLOBAL)
    # Tests whether symmetry controllability is axis-specific
    z_flipped = np.flip(filled, axis=2)
    z_sym_inter = (filled & z_flipped).sum()
    z_sym_union = (filled | z_flipped).sum()
    z_symmetry_iou = float(z_sym_inter / max(z_sym_union, 1))

    # Layer consistency: average IoU between consecutive Y layers (SEMI-LOCAL)
    # Measures vertical structural coherence — "how consistent are the floors?"
    if len(ys) >= 2:
        y_min_l, y_max_l = int(ys[0]), int(ys[-1])
        layer_ious = []
        for y_l in range(y_min_l, y_max_l):
            la = filled[:, y_l, :]
            lb = filled[:, y_l + 1, :]
            l_inter = (la & lb).sum()
            l_union = (la | lb).sum()
            if l_union > 0:
                layer_ious.append(float(l_inter / l_union))
        layer_consistency = float(np.mean(layer_ious)) if layer_ious else 0.0
    else:
        layer_consistency = 1.0

    # Footprint convexity: footprint fill / bounding box footprint area (SEMI-GLOBAL)
    # Measures how "regular/rectangular" the building footprint is
    if len(xs) > 0 and len(zs) > 0:
        xz_proj = filled.any(axis=1)
        fp_area = int(xz_proj.sum())
        bbox_fp = x_span * z_span
        footprint_convexity = fp_area / max(bbox_fp, 1)
    else:
        footprint_convexity = 0.0

    return {
        'height': height,
        'n_blocks': n_blocks,
        'symmetry_iou': round(symmetry_iou, 4),
        'enclosed_ratio': round(enclosed_ratio, 4),
        'elongation': round(elongation, 4),
        'floor_count': floor_count,
        'hollowness': round(hollowness, 4),
        'bbox_volume': bbox_volume,
        'surface_ratio': round(surface_ratio, 4),
        'connected_components': connected_components,
        'vertical_aspect': round(vertical_aspect, 4),
        'z_symmetry_iou': round(z_symmetry_iou, 4),
        'layer_consistency': round(layer_consistency, 4),
        'footprint_convexity': round(footprint_convexity, 4),
    }


def generate_and_measure_all(ar, vqvae, grid_size, spec, seeds, n_per_seed=8, device='cuda', cfg_scale=2.0):
    """Generate across multiple seeds and measure all properties.

    cfg_scale: classifier-free guidance weight. Default 2.0 matches
    inference default.
    """
    all_props = []
    for seed in seeds:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        features = make_features(spec, device, n_per_seed)
        codes = ar.generate_batch(
            batch_size=n_per_seed, struct_features=features,
            temperature=0.9, top_k=100, device=device, cfg_scale=cfg_scale,
        )
        code_grid = codes.reshape(n_per_seed, grid_size, grid_size, grid_size)
        logits = vqvae.decode_from_indices(code_grid)
        voxels_batch = logits.argmax(dim=1).cpu().numpy()
        for v in voxels_batch:
            all_props.append(measure_all_properties(v))
    return all_props


def measure_training_data(n_samples=200):
    """Measure properties on a sample of training builds."""
    import csv
    builds_dir = PROJECT_ROOT / "data" / "processed"
    manifest_path = builds_dir / "manifest.csv"
    remap_path = builds_dir / "token_remap.json"

    with open(remap_path) as f:
        raw_remap = json.load(f)
    max_old = max(int(k) for k in raw_remap.keys())
    remap_table = np.zeros(max_old + 1, dtype=np.int64)
    for old_str, new_id in raw_remap.items():
        remap_table[int(old_str)] = new_id

    with open(manifest_path, 'r', encoding='utf-8') as f:
        entries = list(csv.DictReader(f))

    # Filter same as training
    valid = [m for m in entries
             if int(m['non_air_blocks']) >= 20
             and max(int(m.get('shape_x', 0)), int(m.get('shape_y', 0)), int(m.get('shape_z', 0))) <= 32]

    rng = np.random.RandomState(42)
    sample = rng.choice(len(valid), min(n_samples, len(valid)), replace=False)

    all_props = []
    for idx in sample:
        m = valid[idx]
        path = builds_dir / m['path']
        try:
            data = np.load(path)
            voxels = data['voxels']
            padded = np.zeros((32, 32, 32), dtype=np.int64)
            sx, sy, sz = voxels.shape
            ox = (32 - sx) // 2
            oz = (32 - sz) // 2
            padded[ox:ox+sx, 0:sy, oz:oz+sz] = voxels.astype(np.int64)
            padded[padded >= len(remap_table)] = 0
            padded = remap_table[padded]
            all_props.append(measure_all_properties(padded))
        except Exception:
            continue

    return all_props


def threshold_sweep(values, thresholds):
    """Compute satisfaction rate at each threshold."""
    arr = np.array(values)
    return {round(t, 3): round(float((arr >= t).mean()), 4) for t in thresholds}


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vqvae_ckpt = str(PROJECT_ROOT / "checkpoints" / "vqvae" / "vqvae_step100000.pt")
    ar_ckpt = str(PROJECT_ROOT / "checkpoints" / "ar_cond" / "ar_cond_step80000.pt")

    print("Loading models...")
    vqvae, ar, grid_size = load_models(vqvae_ckpt, ar_ckpt, device)

    seeds = [42, 123, 456, 789, 2026]
    n_per_seed = 8

    # ============================================================
    # Part 1: Generate with different conditions
    # ============================================================
    conditions = {
        'unconditioned': {},
        'height=tower': {'height': 3},
        'size=large': {'size': 3},
        'symmetry=1': {'symmetry': 1},
        'enclosure=1': {'enclosure': 1},
        # Emergent/implicit tests
        'height=flat+size=large': {'height': 0, 'size': 3},
        'height=tower+size=tiny': {'height': 3, 'size': 0},
    }

    cond_results = {}
    for cond_name, spec in conditions.items():
        print(f"\n  Generating: {cond_name}...")
        props = generate_and_measure_all(ar, vqvae, grid_size, spec, seeds, n_per_seed, device)
        cond_results[cond_name] = props

    # ============================================================
    # Part 2: Measure training data
    # ============================================================
    print("\n  Measuring training data (200 samples)...")
    train_props = measure_training_data(200)

    # ============================================================
    # Part 3: Threshold sweeps
    # ============================================================
    print(f"\n{'='*80}")
    print("THRESHOLD SWEEPS")
    print(f"{'='*80}")

    # Symmetry sweep
    sym_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"\nSymmetry IoU threshold sweep (40 samples per condition):")
    print(f"{'Condition':<25} " + " ".join(f">{t:.1f}" for t in sym_thresholds))
    print("-" * 100)
    for cond_name, props in [('training', train_props)] + list(cond_results.items()):
        vals = [p['symmetry_iou'] for p in props]
        rates = threshold_sweep(vals, sym_thresholds)
        rates_str = " ".join(f"{rates[round(t,3)]:5.1%}" for t in sym_thresholds)
        print(f"{cond_name:<25} {rates_str}")

    # Enclosure sweep
    enc_thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    print(f"\nEnclosure ratio threshold sweep:")
    print(f"{'Condition':<25} " + " ".join(f">{t:.3f}" for t in enc_thresholds))
    print("-" * 100)
    for cond_name, props in [('training', train_props)] + list(cond_results.items()):
        vals = [p['enclosed_ratio'] for p in props]
        rates = threshold_sweep(vals, enc_thresholds)
        rates_str = " ".join(f"{rates[round(t,3)]:6.1%}" for t in enc_thresholds)
        print(f"{cond_name:<25} {rates_str}")

    # ============================================================
    # Part 4: New properties — regime placement
    # ============================================================
    print(f"\n{'='*80}")
    print("PROPERTY REGIME MAP")
    print("All 7 properties across conditions (mean +/- std)")
    print(f"{'='*80}")

    prop_names = ['height', 'n_blocks', 'symmetry_iou', 'enclosed_ratio',
                  'elongation', 'floor_count', 'hollowness',
                  'bbox_volume', 'surface_ratio', 'connected_components',
                  'vertical_aspect', 'z_symmetry_iou',
                  'layer_consistency', 'footprint_convexity']

    print(f"\n{'Condition':<25} " + " ".join(f"{'%s'%p:>14}" for p in prop_names))
    print("-" * 130)

    all_stats = {}
    for cond_name, props in [('training', train_props)] + list(cond_results.items()):
        stats = {}
        row_parts = []
        for pname in prop_names:
            vals = [p[pname] for p in props]
            mean = np.mean(vals)
            std = np.std(vals)
            stats[pname] = {'mean': round(mean, 3), 'std': round(std, 3)}
            row_parts.append(f"{mean:>7.2f}+{std:<5.2f}")
        all_stats[cond_name] = stats
        print(f"{cond_name:<25} " + " ".join(row_parts))

    # ============================================================
    # Part 5: Controllability score per property
    # ============================================================
    print(f"\n{'='*80}")
    print("CONTROLLABILITY ANALYSIS")
    print("How much does each property shift when conditioned vs unconditioned?")
    print(f"{'='*80}")

    uncond_stats = all_stats['unconditioned']

    # For each property, find the condition that MOST affects it
    print(f"\n{'Property':<18} {'Uncond Mean':>12} {'Best Cond':>25} {'Cond Mean':>12} {'Shift':>8} {'Shift%':>8}")
    print("-" * 90)

    regime_assignments = {}
    for pname in prop_names:
        uncond_mean = uncond_stats[pname]['mean']
        best_shift = 0
        best_cond = 'none'
        best_cond_mean = uncond_mean

        for cond_name in cond_results:
            if cond_name == 'unconditioned':
                continue
            cond_mean = all_stats[cond_name][pname]['mean']
            shift = abs(cond_mean - uncond_mean)
            if shift > best_shift:
                best_shift = shift
                best_cond = cond_name
                best_cond_mean = cond_mean

        shift_pct = best_shift / max(abs(uncond_mean), 0.001) * 100
        print(f"{pname:<18} {uncond_mean:>12.3f} {best_cond:>25} {best_cond_mean:>12.3f} "
              f"{best_shift:>8.3f} {shift_pct:>7.1f}%")

        # Assign regime
        if shift_pct > 100:
            regime = "CONTROLLABLE"
        elif shift_pct > 20:
            regime = "APPROACHABLE"
        else:
            regime = "UNRESPONSIVE"
        regime_assignments[pname] = regime

    print(f"\n{'='*80}")
    print("REGIME ASSIGNMENTS")
    print(f"{'='*80}")
    for regime in ["CONTROLLABLE", "APPROACHABLE", "UNRESPONSIVE"]:
        props_in = [p for p, r in regime_assignments.items() if r == regime]
        print(f"  {regime}: {', '.join(props_in)}")

    # ============================================================
    # Part 6: Correlation between new and existing properties
    # ============================================================
    print(f"\n{'='*80}")
    print("EMERGENT BEHAVIOR")
    print("Do conditions designed for one property implicitly affect others?")
    print(f"{'='*80}")

    emergent_pairs = [
        ('height=tower', 'floor_count', 'Does tall = more floors?'),
        ('height=flat+size=large', 'elongation', 'Does flat+large = elongated?'),
        ('height=flat+size=large', 'hollowness', 'Does flat+large = hollow?'),
        ('size=large', 'hollowness', 'Does large = more hollow?'),
        ('height=tower+size=tiny', 'elongation', 'Does tower+tiny = elongated?'),
        ('enclosure=1', 'hollowness', 'Does enclosure cond = more hollow?'),
        # New emergent tests
        ('size=large', 'bbox_volume', 'Does large = bigger bbox?'),
        ('height=tower', 'vertical_aspect', 'Does tower = higher vertical aspect?'),
        ('symmetry=1', 'z_symmetry_iou', 'Does X-sym cond also increase Z-sym?'),
        ('height=tower', 'connected_components', 'Does tower = more fragments?'),
        ('size=large', 'surface_ratio', 'Does large = lower surface ratio?'),
        ('height=tower+size=tiny', 'connected_components', 'Does tower+tiny = fragmented?'),
        ('symmetry=1', 'surface_ratio', 'Does symmetry = lower surface ratio?'),
    ]

    for cond_name, prop, question in emergent_pairs:
        uncond_vals = [p[prop] for p in cond_results['unconditioned']]
        cond_vals = [p[prop] for p in cond_results[cond_name]]
        uncond_mean = np.mean(uncond_vals)
        cond_mean = np.mean(cond_vals)
        shift = cond_mean - uncond_mean
        shift_pct = shift / max(abs(uncond_mean), 0.001) * 100
        sig = "YES" if abs(shift_pct) > 30 else "weak" if abs(shift_pct) > 10 else "no"
        print(f"  {question}")
        print(f"    uncond={uncond_mean:.3f}, cond={cond_mean:.3f}, "
              f"shift={shift:+.3f} ({shift_pct:+.1f}%) [{sig}]")

    # Save results
    output = {
        'conditions': {k: [dict(p) for p in v] for k, v in cond_results.items()},
        'training': [dict(p) for p in train_props],
        'stats': all_stats,
        'regime_assignments': regime_assignments,
    }
    out_path = PROJECT_ROOT / "outputs" / "regime_map_results.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=lambda o: int(o) if hasattr(o, 'item') else str(o))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
