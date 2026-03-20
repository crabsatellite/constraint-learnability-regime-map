"""
Regime Predictor: predict constraint learnability from pre-experiment features.

Given measurable properties of a constraint (training frequency, spatial locality,
training variance), predict whether it will be CONTROLLABLE, APPROACHABLE, or
UNRESPONSIVE — WITHOUT running generation experiments.

This transforms the regime map from a post-hoc taxonomy into a predictive framework.
"""

import sys
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def compute_predictor_features():
    """
    Compute pre-experiment features for each of the 7 regime map properties.

    Features:
      1. effective_frequency: How much training data supports the target condition?
         - Direct: % of data with target conditioning value
         - Emergent: max |correlation with conditioned property| * that property's frequency
      2. locality_score: Spatial computation complexity (1=local, 2=semi-global, 3=global)
      3. training_cv: Coefficient of variation in training data
    """
    # Load training features
    with open(PROJECT_ROOT / "data/processed/structural_features.json") as f:
        sf = json.load(f)

    features_dict = sf["features"]
    n_total = sf["total_builds"]
    bucket_dist = sf["bucket_distributions"]

    # Load regime map training samples (subset with full 7-property measurements)
    with open(PROJECT_ROOT / "outputs/regime_map_results.json") as f:
        regime_data = json.load(f)

    training_samples = regime_data["training"]  # 200 samples
    regime_assignments = regime_data["regime_assignments"]

    # ================================================================
    # Training distribution statistics
    # ================================================================
    props = ["height", "n_blocks", "symmetry_iou", "enclosed_ratio",
             "elongation", "floor_count", "hollowness",
             "bbox_volume", "surface_ratio", "connected_components",
             "vertical_aspect", "z_symmetry_iou",
             "layer_consistency", "footprint_convexity"]

    training_stats = {}
    training_arrays = {}
    for p in props:
        vals = np.array([s[p] for s in training_samples])
        training_arrays[p] = vals
        mean = vals.mean()
        std = vals.std()
        cv = std / max(mean, 1e-6)
        training_stats[p] = {"mean": mean, "std": std, "cv": cv}

    # ================================================================
    # Direct conditioning frequency
    # ================================================================
    # Which properties have dedicated conditioning tokens?
    direct_cond_map = {
        "height":        ("height_bucket",    "3",  int(bucket_dist["height_bucket"].get("3", 0))),
        "n_blocks":      ("size_bucket",      "3",  int(bucket_dist["size_bucket"].get("3", 0))),
        "symmetry_iou":  ("symmetry_flag",    "1",  int(bucket_dist["symmetry_flag"].get("1", 0))),
        "enclosed_ratio":("enclosure_flag",   "1",  int(bucket_dist["enclosure_flag"].get("1", 0))),
    }

    # ================================================================
    # Correlations for emergent properties
    # ================================================================
    # For properties WITHOUT direct conditioning, compute correlation with
    # each conditioned property to find "effective signal strength"
    conditioned_props = ["height", "n_blocks", "symmetry_iou", "enclosed_ratio"]
    emergent_props = ["elongation", "floor_count", "hollowness",
                      "bbox_volume", "surface_ratio", "connected_components",
                      "vertical_aspect", "z_symmetry_iou",
                      "layer_consistency", "footprint_convexity"]

    correlations = {}
    for ep in emergent_props:
        correlations[ep] = {}
        for cp in conditioned_props:
            r = np.corrcoef(training_arrays[ep], training_arrays[cp])[0, 1]
            correlations[ep][cp] = r

    # ================================================================
    # Effective frequency for each property
    # ================================================================
    effective_freq = {}

    for p in props:
        if p in direct_cond_map:
            # Direct conditioning: target frequency in training data
            _, _, count = direct_cond_map[p]
            effective_freq[p] = count / n_total
        else:
            # Emergent: max |correlation| * correlated property's frequency
            max_eff = 0.0
            best_proxy = None
            for cp in conditioned_props:
                r = abs(correlations[p][cp])
                cp_freq = direct_cond_map[cp][2] / n_total
                eff = r * cp_freq
                if eff > max_eff:
                    max_eff = eff
                    best_proxy = cp
            # Scale by the max correlation itself (higher correlation = more signal)
            max_corr = max(abs(correlations[p][cp]) for cp in conditioned_props)
            effective_freq[p] = max_corr  # Use correlation magnitude as signal strength
            correlations[p]["_best_proxy"] = best_proxy
            correlations[p]["_max_corr"] = max_corr

    # ================================================================
    # Locality scores
    # ================================================================
    # 1 = LOCAL: computed from per-voxel aggregates (count, max coordinate)
    # 2 = SEMI-GLOBAL: requires slice-level or bounding box statistics
    # 3 = GLOBAL: requires full spatial analysis (BFS, mirror, topology)
    locality = {
        "height":        1,  # max y-coordinate of any non-air voxel
        "n_blocks":      1,  # count of non-air voxels
        "symmetry_iou":  3,  # mirror comparison along full axis
        "enclosed_ratio":3,  # BFS flood fill from boundary faces
        "elongation":    2,  # bounding box dimension ratio
        "floor_count":   2,  # count of horizontal slices with occupancy
        "hollowness":    2,  # fill ratio within bounding box
        "bbox_volume":   1,  # product of bounding box dimensions
        "surface_ratio": 2,  # neighbor checking per voxel
        "connected_components": 3,  # BFS over filled voxels
        "vertical_aspect": 2,  # height / max horizontal span
        "z_symmetry_iou": 3,  # mirror comparison along Z axis
        "layer_consistency": 2,  # adjacent Y-layer IoU comparison
        "footprint_convexity": 2,  # footprint fill ratio in bbox
    }

    # ================================================================
    # Build feature table
    # ================================================================
    regime_map = {"CONTROLLABLE": 0, "APPROACHABLE": 1, "UNRESPONSIVE": 2}

    rows = []
    print("\n" + "=" * 90)
    print("REGIME PREDICTOR: Feature Table")
    print("=" * 90)
    print(f"\n{'Property':<18} {'EffFreq':>8} {'Locality':>8} {'TrainCV':>8} {'HasCond':>8} {'Regime':<15}")
    print("-" * 72)

    for p in props:
        has_cond = 1 if p in direct_cond_map else 0
        row = {
            "property": p,
            "effective_frequency": round(effective_freq[p], 4),
            "locality": locality[p],
            "training_cv": round(training_stats[p]["cv"], 4),
            "has_direct_cond": has_cond,
            "regime": regime_assignments[p],
            "regime_idx": regime_map[regime_assignments[p]],
        }
        rows.append(row)
        print(f"{p:<18} {row['effective_frequency']:>8.4f} {locality[p]:>8} "
              f"{row['training_cv']:>8.4f} {has_cond:>8} {regime_assignments[p]:<15}")

    # Show correlations for emergent properties
    print("\n--- Emergent property correlations ---")
    for ep in emergent_props:
        corrs = {cp: round(correlations[ep][cp], 3) for cp in conditioned_props}
        print(f"  {ep}: {corrs}  (best proxy: {correlations[ep]['_best_proxy']})")

    return rows, correlations


def continuous_analysis(rows):
    """
    Instead of predicting discrete regimes (overfits with n=7),
    show that predictor features CONTINUOUSLY CORRELATE with controllability.

    This is the honest claim: signal strength and training variation jointly
    predict WHERE a constraint falls on the controllability spectrum.
    The three regimes are descriptive labels for regions of this space.
    """
    print("\n" + "=" * 90)
    print("CONTINUOUS CONTROLLABILITY ANALYSIS")
    print("=" * 90)

    regime_names = {0: "CONTROLLABLE", 1: "APPROACHABLE", 2: "UNRESPONSIVE"}

    # ================================================================
    # Continuous controllability scores
    # from the GENERATION EXPERIMENTS (this is the dependent variable)
    # ================================================================
    with open(PROJECT_ROOT / "outputs" / "regime_map_results.json") as f:
        regime_data = json.load(f)

    stats = regime_data["stats"]
    props = [r["property"] for r in rows]

    # Controllability score = max |relative shift| across all conditions
    conditions = [k for k in stats.keys() if k not in ("training", "unconditioned")]
    uncond = stats["unconditioned"]

    controllability_scores = {}
    for p in props:
        max_shift = 0.0
        best_cond = None
        for cond in conditions:
            unc_mean = uncond[p]["mean"]
            cond_mean = stats[cond][p]["mean"]
            if abs(unc_mean) > 1e-6:
                rel_shift = abs(cond_mean - unc_mean) / abs(unc_mean)
            else:
                rel_shift = abs(cond_mean - unc_mean)  # absolute for near-zero
            if rel_shift > max_shift:
                max_shift = rel_shift
                best_cond = cond
        controllability_scores[p] = {"score": max_shift, "best_condition": best_cond}

    print(f"\n{'Property':<18} {'Score':>8} {'Best Condition':<25} {'Regime':<15}")
    print("-" * 70)
    for r in rows:
        p = r["property"]
        cs = controllability_scores[p]
        print(f"{p:<18} {cs['score']:>8.3f} {cs['best_condition']:<25} {r['regime']:<15}")

    # ================================================================
    # Correlate with predictor features
    # ================================================================
    scores = np.array([controllability_scores[r["property"]]["score"] for r in rows])
    eff_freq = np.array([r["effective_frequency"] for r in rows])
    training_cv = np.array([r["training_cv"] for r in rows])
    has_cond = np.array([r["has_direct_cond"] for r in rows])
    locality = np.array([r["locality"] for r in rows])

    print(f"\n--- Rank correlations (Spearman) ---")
    from scipy.stats import spearmanr
    features = {
        "effective_frequency": eff_freq,
        "training_cv": training_cv,
        "has_direct_cond": has_cond,
        "locality": locality,
    }

    for fname, fvals in features.items():
        rho, pval = spearmanr(fvals, scores)
        print(f"  {fname:<25} rho={rho:+.3f}  p={pval:.3f}")

    # ================================================================
    # Composite predictor
    # ================================================================
    # Try a simple composite: signal_strength * training_cv_factor
    # where signal_strength = eff_freq for direct, max|corr| for emergent
    # and cv_factor = min(cv, 1) / 1  (caps at 1, penalizes low variation)

    print(f"\n--- Composite predictor: signal * min(cv, 1) ---")
    composite = eff_freq * np.minimum(training_cv, 1.0)
    rho, pval = spearmanr(composite, scores)
    print(f"  Spearman rho={rho:+.3f}  p={pval:.3f}")

    print(f"\n  {'Property':<18} {'Signal':>8} {'CV':>8} {'Composite':>10} {'Score':>8} {'Regime':<15}")
    print(f"  {'-'*75}")
    # Sort by composite score descending
    sorted_idx = np.argsort(-composite)
    for i in sorted_idx:
        r = rows[i]
        print(f"  {r['property']:<18} {eff_freq[i]:>8.4f} {training_cv[i]:>8.4f} "
              f"{composite[i]:>10.4f} {scores[i]:>8.3f} {r['regime']:<15}")

    # ================================================================
    # Alternative composite: log-scale cv
    # ================================================================
    print(f"\n--- Alternative: signal * log(1 + cv) ---")
    composite2 = eff_freq * np.log1p(training_cv)
    rho2, pval2 = spearmanr(composite2, scores)
    print(f"  Spearman rho={rho2:+.3f}  p={pval2:.3f}")

    # ================================================================
    # Best single predictor per branch
    # ================================================================
    print(f"\n--- Per-branch analysis ---")
    direct_rows = [r for r in rows if r["has_direct_cond"] == 1]
    emergent_rows = [r for r in rows if r["has_direct_cond"] == 0]

    print(f"\n  Direct (n={len(direct_rows)}):")
    d_freq = np.array([r["effective_frequency"] for r in direct_rows])
    d_scores = np.array([controllability_scores[r["property"]]["score"] for r in direct_rows])
    if len(direct_rows) >= 3:
        rho_d, pval_d = spearmanr(d_freq, d_scores)
        print(f"    freq vs controllability: rho={rho_d:+.3f} p={pval_d:.3f}")
    for r in direct_rows:
        p = r["property"]
        print(f"    {p:<18} freq={r['effective_frequency']:.4f}  "
              f"score={controllability_scores[p]['score']:.3f}  {r['regime']}")

    print(f"\n  Emergent (n={len(emergent_rows)}):")
    e_corr = np.array([r["effective_frequency"] for r in emergent_rows])
    e_cv = np.array([r["training_cv"] for r in emergent_rows])
    e_scores = np.array([controllability_scores[r["property"]]["score"] for r in emergent_rows])
    if len(emergent_rows) >= 3:
        rho_ec, pval_ec = spearmanr(e_corr, e_scores)
        rho_ev, pval_ev = spearmanr(e_cv, e_scores)
        print(f"    correlation vs controllability: rho={rho_ec:+.3f} p={pval_ec:.3f}")
        print(f"    cv vs controllability:          rho={rho_ev:+.3f} p={pval_ev:.3f}")
    for r in emergent_rows:
        p = r["property"]
        print(f"    {p:<18} corr={r['effective_frequency']:.4f} cv={r['training_cv']:.4f}  "
              f"score={controllability_scores[p]['score']:.3f}  {r['regime']}")

    return {
        "controllability_scores": controllability_scores,
        "composite_spearman": {"rho": float(rho), "pval": float(pval)},
    }


def hierarchical_tree(rows):
    """
    Hierarchical decision tree that respects the structure of the problem.

    Key insight: direct-conditioned and emergent properties operate under
    fundamentally different mechanisms, so the predictor should split on
    has_direct_cond first, then use the most relevant feature within each branch.

    Direct branch:  frequency threshold -> CONTROLLABLE vs APPROACHABLE
    Emergent branch: training_cv + proxy_correlation -> CONTROLLABLE / APPROACHABLE / UNRESPONSIVE
    """
    print("\n" + "=" * 90)
    print("HIERARCHICAL DECISION TREE")
    print("=" * 90)

    regime_names = {0: "CONTROLLABLE", 1: "APPROACHABLE", 2: "UNRESPONSIVE"}

    direct = [r for r in rows if r["has_direct_cond"] == 1]
    emergent = [r for r in rows if r["has_direct_cond"] == 0]

    print(f"\n--- Branch 1: Directly conditioned ({len(direct)} properties) ---")
    for r in direct:
        print(f"  {r['property']:<18} freq={r['effective_frequency']:.4f}  -> {r['regime']}")

    print(f"\n--- Branch 2: Emergent ({len(emergent)} properties) ---")
    for r in emergent:
        print(f"  {r['property']:<18} corr={r['effective_frequency']:.4f} "
              f"cv={r['training_cv']:.4f}  -> {r['regime']}")

    # ================================================================
    # Direct branch: find frequency threshold
    # ================================================================
    # Sort by frequency: enclosure(0.024) < symmetry(0.039) < height(0.068) < n_blocks(0.117)
    # APPROACHABLE: enclosure(0.024)
    # CONTROLLABLE: symmetry(0.039), height(0.068), n_blocks(0.117)
    # Threshold: between 0.024 and 0.039

    direct_freqs = sorted([(r["effective_frequency"], r["regime_idx"], r["property"]) for r in direct])
    print(f"\n  Direct properties sorted by frequency:")
    for freq, regime_idx, name in direct_freqs:
        print(f"    {name:<18} freq={freq:.4f} -> {regime_names[regime_idx]}")

    # Find optimal threshold
    best_direct = {"acc": 0, "threshold": 0}
    freq_values = [f for f, _, _ in direct_freqs]
    for i in range(len(freq_values) - 1):
        t = (freq_values[i] + freq_values[i + 1]) / 2
        for low_class in range(3):
            for high_class in range(3):
                correct = sum(
                    (high_class if f >= t else low_class) == r
                    for f, r, _ in direct_freqs
                )
                acc = correct / len(direct_freqs)
                if acc > best_direct["acc"]:
                    best_direct = {
                        "acc": acc, "threshold": t,
                        "low": low_class, "high": high_class,
                    }

    print(f"\n  Direct branch rule:")
    print(f"    freq >= {best_direct['threshold']:.4f} -> {regime_names[best_direct['high']]}")
    print(f"    freq <  {best_direct['threshold']:.4f} -> {regime_names[best_direct['low']]}")
    print(f"    Accuracy: {best_direct['acc']:.0%} ({len(direct)} properties)")

    # ================================================================
    # Emergent branch: find best split on cv and/or correlation
    # ================================================================
    # floor_count: corr=0.765, cv=0.758 -> CONTROLLABLE
    # elongation:  corr=0.179, cv=1.762 -> APPROACHABLE
    # hollowness:  corr=0.311, cv=0.226 -> UNRESPONSIVE
    #
    # Key observation: needs 2 thresholds to separate 3 classes
    # Try: cv threshold first, then correlation threshold

    print(f"\n  Emergent branch:")
    emergent_data = [(r["effective_frequency"], r["training_cv"], r["regime_idx"], r["property"])
                     for r in emergent]

    # With only 3 points, try all possible single-feature splits
    best_emergent = {"acc": 0}

    # Strategy: 2-threshold tree on single feature (cv)
    # OR: 1-threshold on each of 2 features
    features_to_try = [
        ("effective_frequency", [r["effective_frequency"] for r in emergent]),
        ("training_cv", [r["training_cv"] for r in emergent]),
    ]
    y_emergent = [r["regime_idx"] for r in emergent]

    # Try single feature with 2 thresholds (depth-2 tree on one axis)
    for fname, fvals in features_to_try:
        sorted_vals = sorted(set(fvals))
        if len(sorted_vals) < 3:
            continue
        for i in range(len(sorted_vals) - 1):
            for j in range(i + 1, len(sorted_vals)):
                t_low = (sorted_vals[i] + sorted_vals[i + 1 if i + 1 < len(sorted_vals) else i]) / 2
                t_high = (sorted_vals[j - 1] + sorted_vals[j]) / 2 if j > 0 else sorted_vals[j] - 0.01
                # Actually just use midpoints between consecutive sorted values
                thresholds = []
                for k in range(len(sorted_vals) - 1):
                    thresholds.append((sorted_vals[k] + sorted_vals[k + 1]) / 2)

                if len(thresholds) < 2:
                    continue

                for t1_idx in range(len(thresholds)):
                    for t2_idx in range(t1_idx + 1, len(thresholds)):
                        t1 = thresholds[t1_idx]
                        t2 = thresholds[t2_idx]
                        # 3 regions: < t1, [t1, t2), >= t2
                        for c_low in range(3):
                            for c_mid in range(3):
                                for c_high in range(3):
                                    pred = []
                                    for v in fvals:
                                        if v < t1:
                                            pred.append(c_low)
                                        elif v < t2:
                                            pred.append(c_mid)
                                        else:
                                            pred.append(c_high)
                                    acc = sum(p == y for p, y in zip(pred, y_emergent)) / len(y_emergent)
                                    if acc > best_emergent["acc"]:
                                        best_emergent = {
                                            "acc": acc, "feature": fname,
                                            "t1": t1, "t2": t2,
                                            "c_low": c_low, "c_mid": c_mid, "c_high": c_high,
                                        }

    # Also try 2-feature model with 1 threshold each
    if len(emergent) >= 3:
        corrs = [r["effective_frequency"] for r in emergent]
        cvs = [r["training_cv"] for r in emergent]
        corr_sorted = sorted(set(corrs))
        cv_sorted = sorted(set(cvs))
        corr_thresholds = [(corr_sorted[i] + corr_sorted[i+1])/2 for i in range(len(corr_sorted)-1)]
        cv_thresholds = [(cv_sorted[i] + cv_sorted[i+1])/2 for i in range(len(cv_sorted)-1)]

        for tc in corr_thresholds:
            for tv in cv_thresholds:
                for q00 in range(3):
                    for q01 in range(3):
                        for q10 in range(3):
                            for q11 in range(3):
                                pred = []
                                for c, v in zip(corrs, cvs):
                                    if c >= tc and v >= tv:
                                        pred.append(q11)
                                    elif c >= tc and v < tv:
                                        pred.append(q10)
                                    elif c < tc and v >= tv:
                                        pred.append(q01)
                                    else:
                                        pred.append(q00)
                                acc = sum(p == y for p, y in zip(pred, y_emergent)) / len(y_emergent)
                                if acc > best_emergent["acc"]:
                                    best_emergent = {
                                        "acc": acc, "feature": "corr_x_cv",
                                        "tc": tc, "tv": tv,
                                        "q00": q00, "q01": q01, "q10": q10, "q11": q11,
                                    }

    if best_emergent.get("feature") == "corr_x_cv":
        tc, tv = best_emergent["tc"], best_emergent["tv"]
        print(f"    Best split: 2-feature (corr x cv)")
        print(f"    corr >= {tc:.4f} AND cv >= {tv:.4f} -> {regime_names[best_emergent['q11']]}")
        print(f"    corr >= {tc:.4f} AND cv <  {tv:.4f} -> {regime_names[best_emergent['q10']]}")
        print(f"    corr <  {tc:.4f} AND cv >= {tv:.4f} -> {regime_names[best_emergent['q01']]}")
        print(f"    corr <  {tc:.4f} AND cv <  {tv:.4f} -> {regime_names[best_emergent['q00']]}")
    else:
        fname = best_emergent["feature"]
        t1, t2 = best_emergent["t1"], best_emergent["t2"]
        print(f"    Best split: {fname}")
        print(f"    {fname} < {t1:.4f}  -> {regime_names[best_emergent['c_low']]}")
        print(f"    {fname} in [{t1:.4f}, {t2:.4f}) -> {regime_names[best_emergent['c_mid']]}")
        print(f"    {fname} >= {t2:.4f} -> {regime_names[best_emergent['c_high']]}")
    print(f"    Accuracy: {best_emergent['acc']:.0%} ({len(emergent)} properties)")

    # ================================================================
    # Combined predictions
    # ================================================================
    print(f"\n--- Full tree predictions ---")
    all_correct = 0
    for r in rows:
        if r["has_direct_cond"]:
            pred = best_direct["high"] if r["effective_frequency"] >= best_direct["threshold"] else best_direct["low"]
        else:
            if best_emergent.get("feature") == "corr_x_cv":
                c, v = r["effective_frequency"], r["training_cv"]
                tc, tv = best_emergent["tc"], best_emergent["tv"]
                if c >= tc and v >= tv:
                    pred = best_emergent["q11"]
                elif c >= tc and v < tv:
                    pred = best_emergent["q10"]
                elif c < tc and v >= tv:
                    pred = best_emergent["q01"]
                else:
                    pred = best_emergent["q00"]
            else:
                val = r[best_emergent["feature"]]
                if val < best_emergent["t1"]:
                    pred = best_emergent["c_low"]
                elif val < best_emergent["t2"]:
                    pred = best_emergent["c_mid"]
                else:
                    pred = best_emergent["c_high"]

        ok = pred == r["regime_idx"]
        all_correct += ok
        status = "OK" if ok else "MISS"
        print(f"  {r['property']:<18} -> {regime_names[pred]:<15} "
              f"(actual: {r['regime']:<15}) [{status}]")

    total_acc = all_correct / len(rows)
    print(f"\nTotal accuracy: {all_correct}/{len(rows)} = {total_acc:.1%}")

    # ================================================================
    # Leave-one-out cross-validation
    # ================================================================
    print(f"\n--- Leave-one-out ---")
    loo_correct = 0
    for i in range(len(rows)):
        held = rows[i]
        train = [r for j, r in enumerate(rows) if j != i]

        t_direct = [r for r in train if r["has_direct_cond"] == 1]
        t_emergent = [r for r in train if r["has_direct_cond"] == 0]

        # Refit direct branch
        if held["has_direct_cond"] == 1 and len(t_direct) >= 2:
            d_freqs = sorted([(r["effective_frequency"], r["regime_idx"]) for r in t_direct])
            best_d = {"acc": 0, "threshold": 0, "low": 0, "high": 0}
            vals = [f for f, _ in d_freqs]
            for k in range(len(vals) - 1):
                t = (vals[k] + vals[k + 1]) / 2
                for lo in range(3):
                    for hi in range(3):
                        c = sum((hi if f >= t else lo) == r for f, r in d_freqs)
                        if c > best_d["acc"] * len(d_freqs):
                            best_d = {"acc": c / len(d_freqs), "threshold": t, "low": lo, "high": hi}
            pred = best_d["high"] if held["effective_frequency"] >= best_d["threshold"] else best_d["low"]

        elif held["has_direct_cond"] == 0 and len(t_emergent) >= 2:
            # Refit emergent branch
            e_corrs = [r["effective_frequency"] for r in t_emergent]
            e_cvs = [r["training_cv"] for r in t_emergent]
            e_y = [r["regime_idx"] for r in t_emergent]

            best_e = {"acc": 0}
            # Try 2-feature split
            cs = sorted(set(e_corrs))
            vs = sorted(set(e_cvs))
            ct = [(cs[k] + cs[k+1])/2 for k in range(len(cs)-1)] if len(cs) > 1 else [cs[0] - 0.01]
            vt = [(vs[k] + vs[k+1])/2 for k in range(len(vs)-1)] if len(vs) > 1 else [vs[0] - 0.01]

            for tc_v in ct:
                for tv_v in vt:
                    for q00 in range(3):
                        for q01 in range(3):
                            for q10 in range(3):
                                for q11 in range(3):
                                    p = []
                                    for c, v in zip(e_corrs, e_cvs):
                                        if c >= tc_v and v >= tv_v:
                                            p.append(q11)
                                        elif c >= tc_v and v < tv_v:
                                            p.append(q10)
                                        elif c < tc_v and v >= tv_v:
                                            p.append(q01)
                                        else:
                                            p.append(q00)
                                    a = sum(pp == yy for pp, yy in zip(p, e_y)) / len(e_y)
                                    if a > best_e["acc"]:
                                        best_e = {"acc": a, "tc": tc_v, "tv": tv_v,
                                                  "q00": q00, "q01": q01, "q10": q10, "q11": q11}

            c, v = held["effective_frequency"], held["training_cv"]
            tc_v, tv_v = best_e["tc"], best_e["tv"]
            if c >= tc_v and v >= tv_v:
                pred = best_e["q11"]
            elif c >= tc_v and v < tv_v:
                pred = best_e["q10"]
            elif c < tc_v and v >= tv_v:
                pred = best_e["q01"]
            else:
                pred = best_e["q00"]
        else:
            # Fallback: majority class from same branch
            same_branch = t_direct if held["has_direct_cond"] else t_emergent
            if same_branch:
                from collections import Counter
                pred = Counter(r["regime_idx"] for r in same_branch).most_common(1)[0][0]
            else:
                pred = 0  # default

        is_correct = pred == held["regime_idx"]
        loo_correct += is_correct
        status = "OK" if is_correct else "MISS"
        print(f"  Hold out {held['property']:<18}: "
              f"predicted={regime_names[pred]:<15} actual={held['regime']:<15} [{status}]")

    loo_acc = loo_correct / len(rows)
    print(f"\n  LOO Accuracy: {loo_correct}/{len(rows)} = {loo_acc:.1%}")

    return {
        "direct_branch": best_direct,
        "emergent_branch": best_emergent,
        "total_accuracy": total_acc,
        "loo_accuracy": loo_acc,
    }


def main():
    rows, correlations = compute_predictor_features()
    cont_result = continuous_analysis(rows)
    tree_result = hierarchical_tree(rows)

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    print(f"""
Hierarchical Decision Tree (3 features, 7 properties):
  Training accuracy: {tree_result['total_accuracy']:.1%}
  LOO accuracy:      {tree_result['loo_accuracy']:.1%}

The regime predictor uses a two-level hierarchy:

  Level 1: Does the property have a DIRECT conditioning signal?

  YES (direct) -> split on TARGET FREQUENCY in training data
    freq >= ~3%  -> CONTROLLABLE
    freq <  ~3%  -> APPROACHABLE

  NO (emergent) -> split on PROXY CORRELATION x TRAINING CV
    high correlation + sufficient CV -> CONTROLLABLE (strong proxy)
    low correlation + high CV        -> APPROACHABLE (variation but no pathway)
    any correlation + low CV         -> UNRESPONSIVE (nothing to learn)

Interpretation:
  Controllability requires BOTH a learning signal AND learnable variation.
  - Direct conditioning provides explicit signal (sufficient freq = controllable)
  - Without direct conditioning, the signal must come from correlation with
    conditioned properties, AND the property must actually vary in training data.
  - If training data is homogeneous for a property (low CV), the model has no
    gradient to learn from, regardless of signal strength.
""")

    # Save results
    output = {
        "feature_table": rows,
        "correlations": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                             for kk, vv in v.items()}
                         for k, v in correlations.items()
                         if k not in ["height", "n_blocks", "symmetry_iou", "enclosed_ratio"]},
        "tree_result": tree_result,
    }

    out_path = PROJECT_ROOT / "outputs" / "regime_predictor_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
