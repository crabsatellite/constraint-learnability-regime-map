"""
Symmetry augmentation for controllability experiment.

For each training build, creates a symmetrized copy:
  - Left half of 8x8x8 latent grid stays
  - Right half = mirror of left half
  - symmetry_flag set to 1 by construction

This is a MINIMAL intervention: only symmetry distribution changes.
All other features (height, size, footprint, enclosure, complexity) are preserved.

Purpose: test whether low-frequency controllability failure is FREQUENCY-limited
(recoverable with more data) vs REPRESENTATION-limited (structural barrier).
"""

import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def symmetrize_latent_grid(grid):
    """Create an X-axis symmetric version of an 8x8x8 latent grid.

    Takes the left half (x=0..3) and mirrors it to the right half (x=4..7).
    Result is guaranteed symmetric along X.
    """
    sym = grid.copy()
    sym[4:, :, :] = grid[3::-1, :, :]  # mirror x=3,2,1,0 -> x=4,5,6,7
    return sym


def main():
    latent_path = PROJECT_ROOT / "data" / "processed" / "latent_codes.npz"
    features_path = PROJECT_ROOT / "data" / "processed" / "structural_features.json"

    # Output paths
    aug_latent_path = PROJECT_ROOT / "data" / "processed" / "latent_codes_sym_aug.npz"
    aug_features_path = PROJECT_ROOT / "data" / "processed" / "structural_features_sym_aug.json"

    # Load originals
    data = np.load(latent_path)
    indices = data['indices']  # (N, 8, 8, 8)
    N = len(indices)
    print(f"Original latent codes: {N}")

    with open(features_path, 'r') as f:
        feat_data = json.load(f)
    features = feat_data['features']
    print(f"Original features: {len(features)} builds")

    # Create symmetrized copies
    sym_indices = np.zeros_like(indices)
    for i in range(N):
        sym_indices[i] = symmetrize_latent_grid(indices[i])

    # Verify symmetry of augmented grids
    n_symmetric = 0
    for i in range(min(100, N)):
        grid = sym_indices[i]
        flipped = np.flip(grid, axis=0)
        if np.array_equal(grid, flipped):
            n_symmetric += 1
    print(f"Symmetry verification (first 100): {n_symmetric}/100 perfectly symmetric")

    # Concatenate: original + symmetrized
    combined_indices = np.concatenate([indices, sym_indices], axis=0)
    print(f"Combined latent codes: {len(combined_indices)} ({N} original + {N} augmented)")

    # Save augmented latent codes
    np.savez_compressed(aug_latent_path, indices=combined_indices)
    print(f"Saved to {aug_latent_path}")

    # Create augmented features
    # For augmented builds: copy all features from original, but set symmetry_flag=1
    aug_features = dict(features)  # copy original
    feature_names = list(features.keys())

    for i, name in enumerate(feature_names):
        aug_name = f"sym_aug_{name}"
        if name in features:
            aug_feat = dict(features[name])
            aug_feat['symmetry_flag'] = 1  # guaranteed symmetric
            # Keep raw values, update symmetry_iou
            if 'raw' in aug_feat:
                aug_feat['raw'] = dict(aug_feat['raw'])
                aug_feat['raw']['symmetry_iou'] = 1.0
            aug_features[aug_name] = aug_feat

    # Recompute distributions
    bucket_distributions = {}
    for key in ['height_bucket', 'size_bucket', 'footprint_bucket',
                'symmetry_flag', 'enclosure_flag', 'complexity_bucket']:
        dist = {}
        for feat in aug_features.values():
            v = str(feat[key])
            dist[v] = dist.get(v, 0) + 1
        bucket_distributions[key] = dist

    aug_data = {
        'features': aug_features,
        'bucket_distributions': bucket_distributions,
        'total_builds': len(aug_features),
        'augmentation': 'symmetry_only',
    }

    with open(aug_features_path, 'w') as f:
        json.dump(aug_data, f)
    print(f"Saved augmented features to {aug_features_path}")

    # Print distribution comparison
    print(f"\n{'='*60}")
    print("Symmetry distribution comparison:")
    print(f"  Original: {feat_data['bucket_distributions']['symmetry_flag']}")
    print(f"  Augmented: {bucket_distributions['symmetry_flag']}")

    orig_sym_pct = int(feat_data['bucket_distributions']['symmetry_flag'].get('1', 0)) / N * 100
    aug_sym_pct = int(bucket_distributions['symmetry_flag'].get('1', 0)) / len(aug_features) * 100
    print(f"  Symmetric ratio: {orig_sym_pct:.1f}% -> {aug_sym_pct:.1f}%")

    # Enclosure should be UNCHANGED (control variable)
    print(f"\nEnclosure distribution (should be unchanged):")
    print(f"  Original: {feat_data['bucket_distributions']['enclosure_flag']}")
    print(f"  Augmented: {bucket_distributions['enclosure_flag']}")


if __name__ == "__main__":
    main()
