"""
CFG Sensitivity Test: Minimal cross-representation validation.

Tests whether the regime map is representation-sensitive by varying
the classifier-free guidance scale. If higher CFG pushes APPROACHABLE
properties toward CONTROLLABLE but UNRESPONSIVE remains fixed, this
confirms the dual bottleneck: frequency vs representation ceiling.

No retraining — only inference-time parameter change.
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.test_regime_map import load_models, make_features, measure_all_properties


def generate_with_cfg(ar, vqvae, grid_size, spec, cfg_scale, seeds, n_per_seed=8, device='cuda'):
    """Generate samples with specified CFG scale."""
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


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vqvae_ckpt = str(PROJECT_ROOT / "checkpoints" / "vqvae" / "vqvae_step100000.pt")
    ar_ckpt = str(PROJECT_ROOT / "checkpoints" / "ar_cond" / "ar_cond_step80000.pt")

    print("Loading models...")
    vqvae, ar, grid_size = load_models(vqvae_ckpt, ar_ckpt, device)

    seeds = [42, 123, 456]
    n_per_seed = 8
    cfg_scales = [0.0, 2.0, 4.0]

    conditions = {
        'unconditioned': {},
        'height=tower': {'height': 3},
        'symmetry=1': {'symmetry': 1},
        'enclosure=1': {'enclosure': 1},
    }

    prop_names = ['height', 'n_blocks', 'symmetry_iou', 'enclosed_ratio',
                  'elongation', 'floor_count', 'hollowness',
                  'surface_ratio', 'connected_components',
                  'layer_consistency', 'footprint_convexity']

    results = {}

    for cfg in cfg_scales:
        print(f"\n--- CFG scale = {cfg} ---")
        results[str(cfg)] = {}
        for cond_name, spec in conditions.items():
            actual_cfg = cfg if cond_name != 'unconditioned' else 0.0
            props = generate_with_cfg(ar, vqvae, grid_size, spec, actual_cfg, seeds, n_per_seed, device)
            results[str(cfg)][cond_name] = props
            n = len(props)
            stats = {}
            for p in prop_names:
                vals = [x[p] for x in props]
                stats[p] = round(np.mean(vals), 4)
            print(f"  {cond_name} (n={n}): " +
                  " ".join(f"{p}={stats[p]:.3f}" for p in ['height', 'symmetry_iou', 'enclosed_ratio', 'hollowness']))

    # Compute controllability scores at each CFG scale
    print(f"\n{'='*90}")
    print(f"CFG SENSITIVITY: Controllability Shift")
    print(f"{'='*90}")

    print(f"\n{'Property':<22} ", end="")
    for cfg in cfg_scales:
        print(f"{'CFG='+str(cfg):>12}", end="")
    print(f"  {'Regime':<15} {'CFG Effect':<15}")
    print("-" * 90)

    regime_at_cfg = {}
    for p in prop_names:
        regime_at_cfg[p] = {}
        print(f"{p:<22} ", end="")
        for cfg in cfg_scales:
            uncond_vals = [x[p] for x in results[str(cfg)]['unconditioned']]
            uncond_mean = np.mean(uncond_vals)

            # Find best conditioned shift
            max_shift_pct = 0.0
            for cond_name in conditions:
                if cond_name == 'unconditioned':
                    continue
                cond_vals = [x[p] for x in results[str(cfg)][cond_name]]
                cond_mean = np.mean(cond_vals)
                shift = abs(cond_mean - uncond_mean)
                shift_pct = shift / max(abs(uncond_mean), 0.001) * 100
                if shift_pct > max_shift_pct:
                    max_shift_pct = shift_pct

            regime_at_cfg[p][str(cfg)] = max_shift_pct
            print(f"{max_shift_pct:>11.1f}%", end="")

        # Determine regime at each CFG and check for transitions
        regimes = []
        for cfg in cfg_scales:
            s = regime_at_cfg[p][str(cfg)]
            if s > 100:
                regimes.append("C")
            elif s > 20:
                regimes.append("A")
            else:
                regimes.append("U")

        regime_str = "/".join(regimes)
        if len(set(regimes)) == 1:
            effect = "STABLE"
        elif regimes[-1] > regimes[0]:
            effect = "AMPLIFIED"
        else:
            effect = "SHIFTED"
        print(f"  {regime_str:<15} {effect:<15}")

    # Summary
    print(f"\n{'='*90}")
    print("CFG SENSITIVITY SUMMARY")
    print(f"{'='*90}")

    stable = []
    amplified = []
    for p in prop_names:
        shifts = [regime_at_cfg[p][str(cfg)] for cfg in cfg_scales]
        if shifts[-1] > shifts[0] * 1.5 and shifts[-1] > 30:
            amplified.append(p)
        else:
            stable.append(p)

    print(f"\n  STABLE (regime unchanged by CFG): {', '.join(stable)}")
    print(f"  AMPLIFIED (stronger with higher CFG): {', '.join(amplified)}")
    print(f"\n  Interpretation:")
    print(f"    If UNRESPONSIVE properties remain stable -> REPRESENTATION CEILING")
    print(f"    If APPROACHABLE properties amplify -> FREQUENCY/SIGNAL BOTTLENECK")

    # Save
    out_path = PROJECT_ROOT / "outputs" / "cfg_sensitivity_results.json"
    output = {
        "cfg_scales": cfg_scales,
        "controllability_by_cfg": regime_at_cfg,
        "stable": stable,
        "amplified": amplified,
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
