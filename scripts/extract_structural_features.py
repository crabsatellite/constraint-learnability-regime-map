"""
Extract structural features from all training builds for AR conditioning.

For each build, computes 6 discretized features:
  1. height_bucket (0-3): vertical extent
  2. size_bucket (0-3): total non-air blocks
  3. footprint_bucket (0-2): XZ bounding box area
  4. symmetry_flag (0-1): bilateral symmetry IoU > 0.7
  5. enclosure_flag (0-1): has enclosed interior air
  6. complexity_bucket (0-2): surface complexity ratio

Output: data/processed/structural_features.json
"""

import csv
import json
import time
import numpy as np
from collections import deque
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).parent.parent


def compute_enclosed_volume(voxels):
    """Flood fill from outside to find enclosed interior air.

    Returns enclosed_air_count, total_air_count.
    """
    shape = voxels.shape
    is_air = (voxels == 0)
    total_air = int(is_air.sum())
    if total_air == 0:
        return 0, 0

    # BFS from all air voxels on the 6 faces
    visited = np.zeros(shape, dtype=bool)
    queue = deque()

    # Seed from all 6 faces
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                if (x == 0 or x == shape[0]-1 or
                    y == 0 or y == shape[1]-1 or
                    z == 0 or z == shape[2]-1):
                    if is_air[x, y, z] and not visited[x, y, z]:
                        visited[x, y, z] = True
                        queue.append((x, y, z))

    # BFS through air voxels
    while queue:
        x, y, z = queue.popleft()
        for dx, dy, dz in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
            nx, ny, nz = x+dx, y+dy, z+dz
            if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
                if is_air[nx, ny, nz] and not visited[nx, ny, nz]:
                    visited[nx, ny, nz] = True
                    queue.append((nx, ny, nz))

    # Enclosed air = air voxels NOT reached by flood fill
    exterior_air = int(visited.sum())
    enclosed_air = total_air - exterior_air
    return enclosed_air, total_air


def compute_symmetry_iou(voxels):
    """Bilateral symmetry: IoU of non-air voxels with X-axis reflection."""
    filled = voxels > 0
    if not filled.any():
        return 0.0
    flipped = np.flip(filled, axis=0)
    intersection = (filled & flipped).sum()
    union = (filled | flipped).sum()
    return float(intersection / max(union, 1))


def compute_surface_area(voxels):
    """Count exposed faces of non-air blocks (6-connectivity)."""
    filled = voxels > 0
    if not filled.any():
        return 0

    exposed = 0
    shape = voxels.shape
    for axis in range(3):
        # Forward direction
        sliced = np.take(filled, range(1, shape[axis]), axis=axis)
        prev = np.take(filled, range(0, shape[axis]-1), axis=axis)
        # Exposed face where filled meets air (or vice versa)
        exposed += int((sliced != prev).sum())
    # Add boundary faces (faces on the edges of the grid)
    for axis in range(3):
        first = np.take(filled, [0], axis=axis)
        last = np.take(filled, [shape[axis]-1], axis=axis)
        exposed += int(first.sum()) + int(last.sum())

    return exposed


def process_build(args):
    """Process a single build file. Called by process pool."""
    path, name = args
    try:
        data = np.load(path)
        voxels = data['voxels']

        # Pad to 32x32x32 (same as training)
        max_dim = 32
        sx, sy, sz = voxels.shape
        if max(sx, sy, sz) > max_dim:
            return name, None

        padded = np.zeros((max_dim, max_dim, max_dim), dtype=np.int64)
        ox = (max_dim - sx) // 2
        oz = (max_dim - sz) // 2
        padded[ox:ox+sx, 0:sy, oz:oz+sz] = voxels.astype(np.int64)

        filled = padded > 0
        non_air = int(filled.sum())
        if non_air < 20:
            return name, None

        # Height (Y axis extent)
        ys = np.where(filled.any(axis=(0, 2)))[0]
        height = int(ys[-1] - ys[0] + 1) if len(ys) > 0 else 0

        # Footprint (XZ bounding box)
        xz_proj = filled.any(axis=1)  # (X, Z)
        xs = np.where(xz_proj.any(axis=1))[0]
        zs = np.where(xz_proj.any(axis=0))[0]
        if len(xs) > 0 and len(zs) > 0:
            footprint_area = int((xs[-1] - xs[0] + 1) * (zs[-1] - zs[0] + 1))
        else:
            footprint_area = 0

        # Symmetry
        symmetry_iou = compute_symmetry_iou(padded)

        # Enclosure
        enclosed_air, total_air = compute_enclosed_volume(padded)
        enclosed_ratio = enclosed_air / max(total_air, 1)

        # Surface complexity
        surface_area = compute_surface_area(padded)
        if non_air > 0:
            min_surface = 6 * (non_air ** (2/3))  # cube with same volume
            surface_complexity = surface_area / max(min_surface, 1)
        else:
            surface_complexity = 0

        # Discretize into buckets
        # Height bucket
        if height <= 8:
            height_bucket = 0
        elif height <= 16:
            height_bucket = 1
        elif height <= 24:
            height_bucket = 2
        else:
            height_bucket = 3

        # Size bucket
        if non_air <= 100:
            size_bucket = 0
        elif non_air <= 500:
            size_bucket = 1
        elif non_air <= 2000:
            size_bucket = 2
        else:
            size_bucket = 3

        # Footprint bucket
        if footprint_area <= 100:
            footprint_bucket = 0
        elif footprint_area <= 400:
            footprint_bucket = 1
        else:
            footprint_bucket = 2

        # Symmetry flag
        symmetry_flag = 1 if symmetry_iou > 0.7 else 0

        # Enclosure flag
        enclosure_flag = 1 if enclosed_ratio > 0.05 else 0

        # Complexity bucket
        if surface_complexity < 1.5:
            complexity_bucket = 0
        elif surface_complexity <= 3.0:
            complexity_bucket = 1
        else:
            complexity_bucket = 2

        return name, {
            'height_bucket': height_bucket,
            'size_bucket': size_bucket,
            'footprint_bucket': footprint_bucket,
            'symmetry_flag': symmetry_flag,
            'enclosure_flag': enclosure_flag,
            'complexity_bucket': complexity_bucket,
            'raw': {
                'height': height,
                'non_air_count': non_air,
                'footprint_area': footprint_area,
                'symmetry_iou': round(symmetry_iou, 4),
                'enclosed_volume_ratio': round(enclosed_ratio, 4),
                'surface_complexity': round(surface_complexity, 4),
            }
        }
    except Exception as e:
        return name, None


def main():
    builds_dir = PROJECT_ROOT / "data" / "processed" / "builds"
    manifest_path = PROJECT_ROOT / "data" / "processed" / "manifest.csv"
    output_path = PROJECT_ROOT / "data" / "processed" / "structural_features.json"

    # Load manifest
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        entries = list(reader)

    print(f"Total manifest entries: {len(entries)}")

    # Prepare build paths
    tasks = []
    for m in entries:
        path = PROJECT_ROOT / "data" / "processed" / m['path']
        name = Path(m['path']).name
        tasks.append((str(path), name))

    print(f"Processing {len(tasks)} builds...")
    start = time.time()

    # Process with multiprocessing
    features = {}
    n_processed = 0
    n_skipped = 0

    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(process_build, t): t for t in tasks}
        for future in as_completed(futures):
            name, result = future.result()
            if result is not None:
                features[name] = result
                n_processed += 1
            else:
                n_skipped += 1

            if (n_processed + n_skipped) % 1000 == 0:
                print(f"  Processed {n_processed + n_skipped}/{len(tasks)}...")

    elapsed = time.time() - start
    print(f"Processed {n_processed} builds, skipped {n_skipped} in {elapsed:.1f}s")

    # Compute bucket distributions
    bucket_distributions = {}
    for key in ['height_bucket', 'size_bucket', 'footprint_bucket',
                'symmetry_flag', 'enclosure_flag', 'complexity_bucket']:
        dist = {}
        for feat in features.values():
            v = str(feat[key])
            dist[v] = dist.get(v, 0) + 1
        bucket_distributions[key] = dist

    # Save
    output = {
        'features': features,
        'bucket_distributions': bucket_distributions,
        'total_builds': n_processed,
    }
    with open(output_path, 'w') as f:
        json.dump(output, f)
    print(f"Saved to {output_path}")

    # Print distribution summary
    print(f"\n{'='*60}")
    print("Feature Distributions:")
    print(f"{'='*60}")
    for key, dist in bucket_distributions.items():
        print(f"\n{key}:")
        total = sum(dist.values())
        for bucket in sorted(dist.keys(), key=int):
            count = dist[bucket]
            pct = 100 * count / total
            bar = '#' * int(pct / 2)
            print(f"  {bucket}: {count:>5d} ({pct:>5.1f}%) {bar}")


if __name__ == "__main__":
    main()
