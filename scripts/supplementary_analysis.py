"""
Supplementary analyses for reviewer concerns:
  S1: Permutation test for composite predictor
  S2: Per-seed enclosed_ratio breakdown + absolute differences
  S3: connected_components confidence assessment
  M1: Threshold sensitivity sweep
  M3: Training data measurement standard errors
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).parent.parent


def load_results():
    with open(PROJECT_ROOT / "outputs" / "regime_map_results.json") as f:
        regime = json.load(f)
    with open(PROJECT_ROOT / "outputs" / "regime_predictor_results.json") as f:
        pred = json.load(f)
    with open(PROJECT_ROOT / "outputs" / "bootstrap_results.json") as f:
        boot = json.load(f)
    return regime, pred, boot


def s1_permutation_test(pred, regime, n_perm=10000, seed=42):
    """Permutation test: how often does random assignment achieve rho >= observed?"""
    print("=" * 70)
    print("S1: PERMUTATION TEST FOR COMPOSITE PREDICTOR")
    print("=" * 70)

    emergent = [p for p in pred['feature_table'] if p['has_direct_cond'] == 0]
    props = [p['property'] for p in emergent]

    # Compute controllability from regime map
    uncond = regime['conditions']['unconditioned']
    ctrl_scores = {}
    for pname in props:
        uncond_mean = np.mean([s[pname] for s in uncond])
        best_shift = 0
        for cname, samples in regime['conditions'].items():
            if cname == 'unconditioned':
                continue
            cmean = np.mean([s[pname] for s in samples])
            shift = abs(cmean - uncond_mean) / max(abs(uncond_mean), 0.001) * 100
            if shift > best_shift:
                best_shift = shift
        ctrl_scores[pname] = best_shift

    # Composite: signal * min(cv, 1)
    composites = []
    ctrls = []
    for p in emergent:
        composites.append(p['effective_frequency'] * min(p['training_cv'], 1.0))
        ctrls.append(ctrl_scores[p['property']])

    composites = np.array(composites)
    ctrls = np.array(ctrls)
    observed_rho, observed_p = spearmanr(composites, ctrls)

    rng = np.random.RandomState(seed)
    count_ge = 0
    for _ in range(n_perm):
        perm_ctrls = rng.permutation(ctrls)
        perm_rho, _ = spearmanr(composites, perm_ctrls)
        if abs(perm_rho) >= abs(observed_rho):
            count_ge += 1

    perm_p = count_ge / n_perm

    print(f"\n  Observed rho: {observed_rho:.4f}")
    print(f"  Parametric p: {observed_p:.4f}")
    print(f"  Permutation p (n={n_perm}): {perm_p:.4f}")
    print(f"  {'SIGNIFICANT' if perm_p < 0.05 else 'NOT SIGNIFICANT'} at alpha=0.05")

    # Also test CV alone
    cvs = np.array([p['training_cv'] for p in emergent])
    obs_cv_rho, obs_cv_p = spearmanr(cvs, ctrls)
    count_cv = sum(1 for _ in range(n_perm)
                   if abs(spearmanr(cvs, rng.permutation(ctrls))[0]) >= abs(obs_cv_rho))
    perm_cv_p = count_cv / n_perm
    print(f"\n  CV alone: rho={obs_cv_rho:.4f}, parametric p={obs_cv_p:.4f}, permutation p={perm_cv_p:.4f}")

    # Proxy correlation alone
    corrs = np.array([p['effective_frequency'] for p in emergent])
    obs_corr_rho, obs_corr_p = spearmanr(corrs, ctrls)
    count_corr = sum(1 for _ in range(n_perm)
                     if abs(spearmanr(corrs, rng.permutation(ctrls))[0]) >= abs(obs_corr_rho))
    perm_corr_p = count_corr / n_perm
    print(f"  Proxy corr alone: rho={obs_corr_rho:.4f}, parametric p={obs_corr_p:.4f}, permutation p={perm_corr_p:.4f}")

    return {
        'composite': {'rho': round(observed_rho, 4), 'parametric_p': round(observed_p, 4),
                       'permutation_p': round(perm_p, 4)},
        'cv_alone': {'rho': round(obs_cv_rho, 4), 'parametric_p': round(obs_cv_p, 4),
                      'permutation_p': round(perm_cv_p, 4)},
        'corr_alone': {'rho': round(obs_corr_rho, 4), 'parametric_p': round(obs_corr_p, 4),
                        'permutation_p': round(perm_corr_p, 4)},
    }


def s2_per_seed_enclosed_ratio(regime):
    """Per-seed breakdown of enclosed_ratio controllability."""
    print(f"\n{'=' * 70}")
    print("S2: PER-SEED ENCLOSED_RATIO ANALYSIS")
    print("=" * 70)

    uncond = regime['conditions']['unconditioned']
    encl = regime['conditions']['enclosure=1']

    # 5 seeds × 8 samples = 40 total
    n_per_seed = 8
    seeds_used = len(uncond) // n_per_seed

    print(f"\n  Seeds: {seeds_used}, samples/seed: {n_per_seed}")
    print(f"\n  {'Seed':>6} {'Uncond Mean':>12} {'Encl=1 Mean':>12} {'Abs Diff':>10} {'Ctrl%':>10}")
    print("  " + "-" * 56)

    seed_ctrls = []
    for s in range(seeds_used):
        start = s * n_per_seed
        end = start + n_per_seed
        u_vals = [uncond[i]['enclosed_ratio'] for i in range(start, end)]
        e_vals = [encl[i]['enclosed_ratio'] for i in range(start, end)]
        u_mean = np.mean(u_vals)
        e_mean = np.mean(e_vals)
        abs_diff = abs(e_mean - u_mean)
        ctrl_pct = abs_diff / max(abs(u_mean), 0.001) * 100
        seed_ctrls.append(ctrl_pct)
        print(f"  {s+1:>6} {u_mean:>12.6f} {e_mean:>12.6f} {abs_diff:>10.6f} {ctrl_pct:>9.1f}%")

    print(f"\n  Mean ctrl%: {np.mean(seed_ctrls):.1f}%")
    print(f"  Std ctrl%:  {np.std(seed_ctrls):.1f}%")
    print(f"  Min/Max:    {np.min(seed_ctrls):.1f}% / {np.max(seed_ctrls):.1f}%")

    # Also report absolute values for all properties with extreme %
    print(f"\n  Absolute differences for high-% properties:")
    for pname in ['enclosed_ratio', 'n_blocks', 'z_symmetry_iou']:
        u_vals = [s[pname] for s in uncond]
        best_abs = 0
        best_cond = ''
        for cname, samples in regime['conditions'].items():
            if cname == 'unconditioned':
                continue
            c_vals = [s[pname] for s in samples]
            diff = abs(np.mean(c_vals) - np.mean(u_vals))
            if diff > best_abs:
                best_abs = diff
                best_cond = cname
        print(f"    {pname}: abs_diff={best_abs:.4f} (best cond: {best_cond}), "
              f"uncond_mean={np.mean(u_vals):.4f}")

    return seed_ctrls


def m1_threshold_sensitivity(regime, boot):
    """Report regime counts under different thresholds."""
    print(f"\n{'=' * 70}")
    print("M1: THRESHOLD SENSITIVITY")
    print("=" * 70)

    # Get point estimates
    props_data = boot['properties']
    ctrl_pcts = {p: d['controllability_pct'] for p, d in props_data.items()}

    thresholds = [
        (10, 80), (15, 90), (20, 100), (25, 110), (30, 120),
    ]

    print(f"\n  {'U_thresh':>8} {'C_thresh':>8} | {'C':>3} {'A':>3} {'U':>3} | Changed properties")
    print("  " + "-" * 65)

    baseline_regimes = {}
    for p, pct in ctrl_pcts.items():
        if pct > 100:
            baseline_regimes[p] = 'C'
        elif pct > 20:
            baseline_regimes[p] = 'A'
        else:
            baseline_regimes[p] = 'U'

    for u_t, c_t in thresholds:
        regimes = {}
        for p, pct in ctrl_pcts.items():
            if pct > c_t:
                regimes[p] = 'C'
            elif pct > u_t:
                regimes[p] = 'A'
            else:
                regimes[p] = 'U'

        n_c = sum(1 for r in regimes.values() if r == 'C')
        n_a = sum(1 for r in regimes.values() if r == 'A')
        n_u = sum(1 for r in regimes.values() if r == 'U')

        changed = [p for p in regimes if regimes[p] != baseline_regimes[p]]
        marker = " <-- baseline" if u_t == 20 and c_t == 100 else ""
        print(f"  {u_t:>8} {c_t:>8} | {n_c:>3} {n_a:>3} {n_u:>3} | {', '.join(changed) if changed else '(none)'}{marker}")

    # Also: CI-conservative classification
    print(f"\n  CI-conservative classification (use CI lower bound for C, upper for U):")
    n_conf_c = sum(1 for d in props_data.values() if d['ci_low'] > 100)
    n_conf_u = sum(1 for d in props_data.values() if d['ci_high'] < 20)
    n_borderline = 14 - n_conf_c - n_conf_u
    print(f"    Confident CONTROLLABLE: {n_conf_c}")
    print(f"    Confident UNRESPONSIVE: {n_conf_u}")
    print(f"    Borderline: {n_borderline}")
    borderline = [p for p, d in props_data.items()
                  if not (d['ci_low'] > 100 or d['ci_high'] < 20)]
    for p in borderline:
        d = props_data[p]
        print(f"      {p}: {d['controllability_pct']:.1f}% [{d['ci_low']:.1f}%, {d['ci_high']:.1f}%]")


def m3_training_se(regime):
    """Standard errors of training data measurements."""
    print(f"\n{'=' * 70}")
    print("M3: TRAINING DATA MEASUREMENT STANDARD ERRORS")
    print("=" * 70)

    train = regime['training']
    n = len(train)
    print(f"\n  Training samples: {n}")

    prop_names = list(train[0].keys())
    print(f"\n  {'Property':<22} {'Mean':>10} {'Std':>10} {'SE':>10} {'CV':>8} {'SE/Mean':>8}")
    print("  " + "-" * 72)
    for p in prop_names:
        vals = [s[p] for s in train]
        mean = np.mean(vals)
        std = np.std(vals)
        se = std / np.sqrt(n)
        cv = std / max(abs(mean), 0.001)
        se_pct = se / max(abs(mean), 0.001) * 100
        print(f"  {p:<22} {mean:>10.4f} {std:>10.4f} {se:>10.4f} {cv:>8.4f} {se_pct:>7.1f}%")


def main():
    regime, pred, boot = load_results()

    perm_results = s1_permutation_test(pred, regime)
    seed_ctrls = s2_per_seed_enclosed_ratio(regime)
    m1_threshold_sensitivity(regime, boot)
    m3_training_se(regime)

    # Save all supplementary results
    output = {
        'permutation_test': perm_results,
        'enclosed_ratio_per_seed_ctrl': [round(x, 2) for x in seed_ctrls],
    }
    out_path = PROJECT_ROOT / "outputs" / "supplementary_results.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
