"""
Composability and OOD generalization tests for structural conditioning.

Experiment 1 (Composability): Are control axes independent?
  - Generate with single axis constraints
  - Generate with combined constraints
  - Measure: does adding axis B change the distribution of axis A?

Experiment 2 (OOD Generalization): Does control work on rare/unseen combinations?
  - Find rare combinations in training data
  - Generate with those combinations
  - Measure constraint satisfaction rate vs common combinations

These experiments prove whether the conditioning forms a TRUE structural interface
(decomposable, generalizable) vs just a learned correlation.
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path

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
    """Create features tensor from partial spec. Unspecified = unconditional."""
    vocabs = ARTransformer3D.STRUCT_FEATURE_VOCABS
    names = ARTransformer3D.STRUCT_FEATURE_NAMES
    uncond = [v for v in vocabs]
    feat = list(uncond)
    for k, v in spec.items():
        idx = names.index(k)
        feat[idx] = v
    return torch.tensor([feat] * n, dtype=torch.long, device=device)


def generate_and_measure(ar, vqvae, grid_size, spec, n=8, device='cuda'):
    """Generate samples with given spec and measure structural properties."""
    features = make_features(spec, device, n)
    codes = ar.generate_batch(
        batch_size=n, struct_features=features,
        temperature=0.9, top_k=100, device=device, cfg_scale=0.0,
    )
    code_grid = codes.reshape(n, grid_size, grid_size, grid_size)
    logits = vqvae.decode_from_indices(code_grid)
    voxels = logits.argmax(dim=1).cpu().numpy()

    measurements = []
    for v in voxels:
        filled = v > 0
        n_blocks = int(filled.sum())
        if n_blocks == 0:
            measurements.append({'blocks': 0, 'height': 0, 'footprint': 0,
                                 'symmetry_iou': 0, 'enclosed_ratio': 0, 'types': 0})
            continue

        # Height
        ys = np.where(filled.any(axis=(0, 2)))[0]
        height = int(ys[-1] - ys[0] + 1) if len(ys) > 0 else 0

        # Footprint area
        xz = filled.any(axis=1)
        xs = np.where(xz.any(axis=1))[0]
        zs = np.where(xz.any(axis=0))[0]
        footprint = int((xs[-1]-xs[0]+1) * (zs[-1]-zs[0]+1)) if len(xs) > 0 and len(zs) > 0 else 0

        # Symmetry IoU
        flipped = np.flip(filled, axis=0)
        inter = (filled & flipped).sum()
        union = (filled | flipped).sum()
        sym_iou = float(inter / max(union, 1))

        # Enclosed volume (simplified: check if any interior air exists)
        from collections import deque
        is_air = ~filled
        visited = np.zeros_like(is_air)
        queue = deque()
        shape = v.shape
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
        total_air = int(is_air.sum())
        enclosed_air = total_air - int(visited.sum())
        enclosed_ratio = enclosed_air / max(total_air, 1)

        types = len(np.unique(v[filled]))
        measurements.append({
            'blocks': n_blocks, 'height': height, 'footprint': footprint,
            'symmetry_iou': round(sym_iou, 4), 'enclosed_ratio': round(enclosed_ratio, 4),
            'types': types,
        })

    return measurements


def bucket_height(h):
    if h <= 8: return 0
    if h <= 16: return 1
    if h <= 24: return 2
    return 3

def bucket_size(n):
    if n <= 100: return 0
    if n <= 500: return 1
    if n <= 2000: return 2
    return 3

def bucket_footprint(a):
    if a <= 100: return 0
    if a <= 400: return 1
    return 2


def experiment_1_composability(ar, vqvae, grid_size, device):
    """Test whether control axes are independently composable."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Axis Composability")
    print("Are constraints independent? Adding axis B should NOT change axis A.")
    print("=" * 70)

    n = 8  # samples per condition

    # Baseline: unconditioned
    uncond = generate_and_measure(ar, vqvae, grid_size, {}, n, device)
    uncond_heights = [m['height'] for m in uncond]
    uncond_blocks = [m['blocks'] for m in uncond]
    print(f"\n  Baseline (uncond): height={np.mean(uncond_heights):.1f}, blocks={np.mean(uncond_blocks):.0f}")

    # Single-axis generations
    single_results = {}
    single_axes = [
        ('height', 3),    # tower
        ('size', 3),      # large
        ('footprint', 2), # sprawling
        ('symmetry', 1),  # symmetric
    ]

    for axis_name, axis_val in single_axes:
        m = generate_and_measure(ar, vqvae, grid_size, {axis_name: axis_val}, n, device)
        single_results[axis_name] = m
        heights = [x['height'] for x in m]
        blocks = [x['blocks'] for x in m]
        syms = [x['symmetry_iou'] for x in m]
        print(f"  {axis_name}={axis_val}: height={np.mean(heights):.1f}, "
              f"blocks={np.mean(blocks):.0f}, sym={np.mean(syms):.3f}")

    # Pairwise combinations
    print(f"\n  --- Pairwise Combinations ---")
    pairs = [
        ('height+size', {'height': 3, 'size': 3}),
        ('height+symmetry', {'height': 3, 'symmetry': 1}),
        ('size+footprint', {'size': 3, 'footprint': 2}),
        ('size+symmetry', {'size': 3, 'symmetry': 1}),
        ('height+footprint', {'height': 3, 'footprint': 2}),
    ]

    composability_results = []
    for pair_name, spec in pairs:
        m = generate_and_measure(ar, vqvae, grid_size, spec, n, device)
        heights = [x['height'] for x in m]
        blocks = [x['blocks'] for x in m]
        syms = [x['symmetry_iou'] for x in m]
        print(f"  {pair_name}: height={np.mean(heights):.1f}, "
              f"blocks={np.mean(blocks):.0f}, sym={np.mean(syms):.3f}")

        # Check composability: does each axis maintain its effect?
        axis_names = pair_name.split('+')
        for ax in axis_names:
            single_m = single_results.get(ax, uncond)
            if ax == 'height':
                single_val = np.mean([x['height'] for x in single_m])
                combo_val = np.mean(heights)
                metric = 'height'
            elif ax == 'size':
                single_val = np.mean([x['blocks'] for x in single_m])
                combo_val = np.mean(blocks)
                metric = 'blocks'
            elif ax == 'symmetry':
                single_val = np.mean([x['symmetry_iou'] for x in single_m])
                combo_val = np.mean(syms)
                metric = 'symmetry'
            elif ax == 'footprint':
                single_val = np.mean([x['footprint'] for x in single_m])
                combo_val = np.mean([x['footprint'] for x in m])
                metric = 'footprint'
            else:
                continue

            drift = abs(combo_val - single_val) / max(abs(single_val), 1)
            composability_results.append({
                'pair': pair_name,
                'axis': ax,
                'metric': metric,
                'single': single_val,
                'combined': combo_val,
                'drift_pct': round(drift * 100, 1),
            })

    print(f"\n  --- Composability Drift Analysis ---")
    print(f"  {'Pair':<25} {'Axis':<12} {'Single':>8} {'Combined':>10} {'Drift%':>8}")
    print(f"  {'-'*65}")
    for r in composability_results:
        drift_marker = " !!!" if r['drift_pct'] > 50 else ""
        print(f"  {r['pair']:<25} {r['axis']:<12} {r['single']:>8.1f} "
              f"{r['combined']:>10.1f} {r['drift_pct']:>7.1f}%{drift_marker}")

    avg_drift = np.mean([r['drift_pct'] for r in composability_results])
    print(f"\n  Average drift: {avg_drift:.1f}%")
    if avg_drift < 30:
        print(f"  [PASS] Axes are reasonably independent (drift < 30%)")
    elif avg_drift < 60:
        print(f"  [PARTIAL] Some axis interference (drift 30-60%)")
    else:
        print(f"  [FAIL] Axes strongly interfere (drift > 60%)")

    return composability_results


def experiment_2_ood_generalization(ar, vqvae, grid_size, device):
    """Test constraint satisfaction on rare/OOD combinations."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: OOD Generalization")
    print("Do constraints work on rare combinations not well-represented in training?")
    print("=" * 70)

    # Training data distribution (from structural_features.json)
    # symmetry=1: 4%, enclosure=1: 2%, complexity=0: 2%, height=3: 7%

    n = 8
    experiments = []

    # Common combinations (well-represented)
    common = [
        ("medium_height + medium_size", {'height': 1, 'size': 2},
         "height=1 is 39%, size=2 is 41%"),
        ("flat + small", {'height': 0, 'size': 1},
         "height=0 is 40%, size=1 is 38%"),
    ]

    # Rare single-axis (underrepresented)
    rare_single = [
        ("symmetric (4% in train)", {'symmetry': 1},
         "only 397/10310 builds"),
        ("enclosed (2% in train)", {'enclosure': 1},
         "only 250/10310 builds"),
        ("blocky (2% in train)", {'complexity': 0},
         "only 247/10310 builds"),
    ]

    # OOD combinations (likely <0.5% co-occurrence)
    ood = [
        ("tower + symmetric + enclosed", {'height': 3, 'symmetry': 1, 'enclosure': 1},
         "tower(7%) x sym(4%) x enc(2%) ~ 0.006%"),
        ("tiny + tower", {'size': 0, 'height': 3},
         "tiny(9%) x tower(7%) = 0.6%, structurally contradictory"),
        ("large + flat + blocky", {'size': 3, 'height': 0, 'complexity': 0},
         "large(12%) x flat(40%) x blocky(2%) ~ 0.1%"),
        ("symmetric + enclosed + detailed", {'symmetry': 1, 'enclosure': 1, 'complexity': 2},
         "sym(4%) x enc(2%) x det(36%) ~ 0.03%"),
    ]

    for category_name, tests in [("COMMON", common), ("RARE_SINGLE", rare_single), ("OOD", ood)]:
        print(f"\n  --- {category_name} ---")
        for name, spec, note in tests:
            m = generate_and_measure(ar, vqvae, grid_size, spec, n, device)

            # Measure constraint satisfaction
            satisfactions = {}
            for axis, val in spec.items():
                if axis == 'height':
                    actual = [bucket_height(x['height']) for x in m]
                    sat = sum(1 for a in actual if a == val) / n
                    satisfactions[axis] = (sat, [x['height'] for x in m])
                elif axis == 'size':
                    actual = [bucket_size(x['blocks']) for x in m]
                    sat = sum(1 for a in actual if a == val) / n
                    satisfactions[axis] = (sat, [x['blocks'] for x in m])
                elif axis == 'footprint':
                    actual = [bucket_footprint(x['footprint']) for x in m]
                    sat = sum(1 for a in actual if a == val) / n
                    satisfactions[axis] = (sat, [x['footprint'] for x in m])
                elif axis == 'symmetry':
                    actual = [1 if x['symmetry_iou'] > 0.7 else 0 for x in m]
                    sat = sum(1 for a in actual if a == val) / n
                    satisfactions[axis] = (sat, [round(x['symmetry_iou'], 2) for x in m])
                elif axis == 'enclosure':
                    actual = [1 if x['enclosed_ratio'] > 0.05 else 0 for x in m]
                    sat = sum(1 for a in actual if a == val) / n
                    satisfactions[axis] = (sat, [round(x['enclosed_ratio'], 3) for x in m])
                elif axis == 'complexity':
                    # Approximate: types count as complexity proxy
                    satisfactions[axis] = (None, [x['types'] for x in m])

            avg_sat = np.mean([s[0] for s in satisfactions.values() if s[0] is not None])
            print(f"\n  {name}  ({note})")
            for axis, (sat, vals) in satisfactions.items():
                sat_str = f"{sat:.0%}" if sat is not None else "N/A"
                print(f"    {axis}: satisfaction={sat_str}, values={vals}")
            print(f"    >> Overall satisfaction: {avg_sat:.0%}")

            experiments.append({
                'name': name,
                'category': category_name,
                'spec': spec,
                'avg_satisfaction': round(avg_sat, 3) if not np.isnan(avg_sat) else 0,
                'measurements': m,
            })

    # Summary comparison
    print(f"\n  --- Satisfaction Rate Summary ---")
    for cat in ["COMMON", "RARE_SINGLE", "OOD"]:
        cat_exps = [e for e in experiments if e['category'] == cat]
        avg = np.mean([e['avg_satisfaction'] for e in cat_exps])
        print(f"  {cat}: avg satisfaction = {avg:.0%}")

    common_sat = np.mean([e['avg_satisfaction'] for e in experiments if e['category'] == "COMMON"])
    ood_sat = np.mean([e['avg_satisfaction'] for e in experiments if e['category'] == "OOD"])
    gap = common_sat - ood_sat
    print(f"\n  Common-OOD gap: {gap:.0%}")
    if gap < 0.15:
        print(f"  [PASS] Model generalizes well to OOD combinations (gap < 15%)")
    elif gap < 0.30:
        print(f"  [PARTIAL] Some generalization degradation (gap 15-30%)")
    else:
        print(f"  [FAIL] Poor OOD generalization (gap > 30%)")

    return experiments


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vqvae_ckpt = str(PROJECT_ROOT / "checkpoints" / "vqvae" / "vqvae_step100000.pt")

    ar_cond_dir = PROJECT_ROOT / "checkpoints" / "ar_cond"
    ckpts = sorted(ar_cond_dir.glob("ar_cond_step*.pt"),
                   key=lambda p: int(p.stem.split('step')[-1]))
    if not ckpts:
        print("No conditioned AR checkpoints found!")
        return
    ar_ckpt = str(ckpts[-1])
    step = int(ckpts[-1].stem.split('step')[-1])
    print(f"Testing checkpoint: {ar_ckpt} (step {step})")

    vqvae, ar, grid_size = load_models(vqvae_ckpt, ar_ckpt, device)

    results_1 = experiment_1_composability(ar, vqvae, grid_size, device)
    results_2 = experiment_2_ood_generalization(ar, vqvae, grid_size, device)

    # Save all results
    output = {
        'checkpoint': ar_ckpt,
        'step': step,
        'composability': results_1,
        'ood_generalization': [
            {k: v for k, v in e.items() if k != 'measurements'}
            for e in results_2
        ],
    }
    out_path = PROJECT_ROOT / "outputs" / "composability_ood_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
