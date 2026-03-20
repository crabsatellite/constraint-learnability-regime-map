#!/usr/bin/env python3
"""
Structural Evaluation Metrics for 3D Voxel Building Generation
==============================================================

A comprehensive evaluation protocol for assessing the quality of generated
32x32x32 voxel buildings. Computes geometric, structural coherence, symmetry,
and diversity metrics, then compares generated outputs against training data.

Metric Categories:
    1. Geometric       — volume enclosure, surface complexity, height/footprint usage
    2. Coherence       — wall continuity, connectedness, vertical support
    3. Symmetry        — bilateral symmetry along X and Z axes
    4. Diversity       — block type variety, entropy, spatial clustering

Usage:
    python eval_structural.py
    python eval_structural.py --generated_dir outputs/generated/ --training_dir data/processed/builds/
    python eval_structural.py --max_training_samples 1000 --output_dir outputs/eval/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import deque
from dataclasses import dataclass, asdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import label


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRID_SIZE = 32
AIR = 0

# 6-connectivity offsets (faces of a cube)
NEIGHBORS_6 = [
    (-1, 0, 0), (1, 0, 0),
    (0, -1, 0), (0, 1, 0),
    (0, 0, -1), (0, 0, 1),
]

# 4 horizontal neighbors (XZ plane)
NEIGHBORS_4_HORIZ = [
    (-1, 0, 0), (1, 0, 0),
    (0, 0, -1), (0, 0, 1),
]


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class StructuralMetrics:
    """All metrics for a single voxel build."""
    # Identity
    filename: str = ""

    # Geometric
    enclosed_volume_ratio: float = 0.0
    surface_complexity: float = 0.0
    height_utilization: float = 0.0
    footprint_density: float = 0.0

    # Structural coherence
    wall_continuity: float = 0.0
    largest_connected_component_ratio: float = 0.0
    vertical_support_ratio: float = 0.0

    # Symmetry
    bilateral_symmetry_x: float = 0.0
    bilateral_symmetry_z: float = 0.0
    best_symmetry: float = 0.0

    # Diversity
    block_type_count: int = 0
    block_type_entropy: float = 0.0
    spatial_block_clustering: float = 0.0

    # Meta
    total_non_air: int = 0
    total_air: int = 0

    def metric_names(self) -> List[str]:
        """Return ordered list of metric field names (excludes identity/meta)."""
        return [
            "enclosed_volume_ratio",
            "surface_complexity",
            "height_utilization",
            "footprint_density",
            "wall_continuity",
            "largest_connected_component_ratio",
            "vertical_support_ratio",
            "bilateral_symmetry_x",
            "bilateral_symmetry_z",
            "best_symmetry",
            "block_type_count",
            "block_type_entropy",
            "spatial_block_clustering",
        ]


# ---------------------------------------------------------------------------
# Geometric Metrics
# ---------------------------------------------------------------------------

def compute_enclosed_volume_ratio(grid: np.ndarray) -> float:
    """
    Flood fill from outside the structure through all 6 boundary faces.
    Interior air voxels unreachable by the flood are 'enclosed'.

    Returns:
        Ratio of enclosed air voxels to total air voxels. Range [0, 1].
    """
    sx, sy, sz = grid.shape
    solid = grid != AIR
    total_air = int(np.sum(~solid))
    if total_air == 0:
        return 1.0  # fully solid, trivially enclosed

    # BFS flood fill from all boundary air voxels
    visited = np.zeros_like(solid, dtype=bool)
    queue = deque()

    # Seed from all 6 faces
    for x in range(sx):
        for y in range(sy):
            for z in range(sz):
                if (x == 0 or x == sx - 1 or
                    y == 0 or y == sy - 1 or
                    z == 0 or z == sz - 1):
                    if not solid[x, y, z] and not visited[x, y, z]:
                        visited[x, y, z] = True
                        queue.append((x, y, z))

    while queue:
        cx, cy, cz = queue.popleft()
        for dx, dy, dz in NEIGHBORS_6:
            nx, ny, nz = cx + dx, cy + dy, cz + dz
            if 0 <= nx < sx and 0 <= ny < sy and 0 <= nz < sz:
                if not solid[nx, ny, nz] and not visited[nx, ny, nz]:
                    visited[nx, ny, nz] = True
                    queue.append((nx, ny, nz))

    exterior_air = int(np.sum(visited))
    enclosed_air = total_air - exterior_air

    if total_air == 0:
        return 0.0
    return enclosed_air / total_air


def compute_surface_complexity(grid: np.ndarray) -> float:
    """
    Surface area (exposed faces of non-air blocks) divided by the minimum
    possible surface area for a solid of the same volume (a cube).

    A perfect cube of volume V has surface area 6 * V^(2/3).
    Higher values indicate more complex, articulated shapes.

    Returns:
        Ratio >= 1.0. Returns 0.0 if no non-air blocks.
    """
    sx, sy, sz = grid.shape
    solid = grid != AIR
    volume = int(np.sum(solid))
    if volume == 0:
        return 0.0

    # Count exposed faces: for each solid voxel, count neighbors that are
    # air or out-of-bounds
    exposed_faces = 0

    # Pad with air (False) on all sides for boundary handling
    padded = np.pad(solid, 1, mode="constant", constant_values=False)

    for dx, dy, dz in NEIGHBORS_6:
        # Shift the padded array and compare
        shifted = padded[1 + dx : sx + 1 + dx,
                         1 + dy : sy + 1 + dy,
                         1 + dz : sz + 1 + dz]
        exposed_faces += int(np.sum(solid & ~shifted))

    # Minimum surface area for a cube of the same volume
    min_surface = 6.0 * (volume ** (2.0 / 3.0))

    if min_surface == 0:
        return 0.0
    return exposed_faces / min_surface


def compute_height_utilization(grid: np.ndarray) -> float:
    """
    Vertical span of non-air blocks relative to grid height.

    Returns:
        (max_y - min_y + 1) / GRID_SIZE. Range (0, 1]. Returns 0 if empty.
    """
    solid = grid != AIR
    y_occupied = np.any(solid, axis=(0, 2))  # collapse X and Z
    if not np.any(y_occupied):
        return 0.0

    y_indices = np.where(y_occupied)[0]
    span = y_indices[-1] - y_indices[0] + 1
    return span / grid.shape[1]


def compute_footprint_density(grid: np.ndarray) -> float:
    """
    Fraction of the XZ bounding box that has at least one non-air block
    in any Y layer.

    Returns:
        Occupied XZ cells / bounding box XZ area. Range (0, 1]. Returns 0 if empty.
    """
    solid = grid != AIR
    # Project onto XZ: any non-air in the Y column
    xz_proj = np.any(solid, axis=1)  # shape (X, Z)
    if not np.any(xz_proj):
        return 0.0

    x_occ = np.any(xz_proj, axis=1)
    z_occ = np.any(xz_proj, axis=0)

    x_indices = np.where(x_occ)[0]
    z_indices = np.where(z_occ)[0]

    bbox_x = x_indices[-1] - x_indices[0] + 1
    bbox_z = z_indices[-1] - z_indices[0] + 1
    bbox_area = bbox_x * bbox_z

    filled = int(np.sum(xz_proj))
    return filled / bbox_area


# ---------------------------------------------------------------------------
# Structural Coherence Metrics
# ---------------------------------------------------------------------------

def compute_wall_continuity(grid: np.ndarray) -> float:
    """
    For each non-air block on the surface (has at least one exposed face),
    count how many of its 4 horizontal neighbors are also non-air surface blocks.
    Average that ratio across all surface blocks.

    Measures wall smoothness: higher values indicate continuous walls rather
    than scattered surface voxels.

    Returns:
        Mean horizontal neighbor ratio for surface blocks. Range [0, 1].
    """
    sx, sy, sz = grid.shape
    solid = grid != AIR

    # Identify surface blocks: solid blocks with at least one air neighbor
    padded = np.pad(solid, 1, mode="constant", constant_values=False)
    is_surface = np.zeros_like(solid, dtype=bool)

    for dx, dy, dz in NEIGHBORS_6:
        neighbor = padded[1 + dx : sx + 1 + dx,
                          1 + dy : sy + 1 + dy,
                          1 + dz : sz + 1 + dz]
        is_surface |= (solid & ~neighbor)

    surface_coords = np.argwhere(is_surface)
    if len(surface_coords) == 0:
        return 0.0

    # For each surface block, count horizontal neighbors that are also surface
    total_ratio = 0.0
    for x, y, z in surface_coords:
        neighbors_present = 0
        neighbors_checked = 0
        for dx, _dy, dz in NEIGHBORS_4_HORIZ:
            nx, nz = x + dx, z + dz
            if 0 <= nx < sx and 0 <= nz < sz:
                neighbors_checked += 1
                if is_surface[nx, y, nz]:
                    neighbors_present += 1
        if neighbors_checked > 0:
            total_ratio += neighbors_present / neighbors_checked

    return total_ratio / len(surface_coords)


def compute_largest_connected_component_ratio(grid: np.ndarray) -> float:
    """
    Ratio of the largest 6-connected component of non-air blocks to total
    non-air blocks. Uses scipy.ndimage.label for efficient labeling.

    Returns:
        Size of largest component / total non-air. Range (0, 1]. Returns 0 if empty.
    """
    solid = grid != AIR
    total = int(np.sum(solid))
    if total == 0:
        return 0.0

    # 6-connectivity structuring element
    struct = np.zeros((3, 3, 3), dtype=int)
    struct[1, 1, 1] = 1
    for dx, dy, dz in NEIGHBORS_6:
        struct[1 + dx, 1 + dy, 1 + dz] = 1

    labeled, num_features = label(solid.astype(int), structure=struct)

    if num_features == 0:
        return 0.0

    # Find largest component
    component_sizes = np.bincount(labeled.ravel())
    # Index 0 is background (air), skip it
    if len(component_sizes) <= 1:
        return 0.0
    largest = int(np.max(component_sizes[1:]))
    return largest / total


def compute_vertical_support_ratio(grid: np.ndarray) -> float:
    """
    For each non-air block above the ground layer (y > 0), check whether the
    block directly below (y-1) is also non-air.

    Measures physical plausibility: floating blocks score low.

    Returns:
        Ratio of supported above-ground blocks. Range [0, 1].
        Returns 1.0 if all blocks are on the ground, 0.0 if no non-air blocks.
    """
    solid = grid != AIR
    # Blocks above ground: y >= 1
    above_ground = solid[:, 1:, :]
    total_above = int(np.sum(above_ground))
    if total_above == 0:
        # All blocks on ground floor or no blocks at all
        return 1.0 if np.any(solid) else 0.0

    # Check support: block at (x, y, z) supported if (x, y-1, z) is solid
    below = solid[:, :-1, :]  # y from 0 to sy-2
    supported = above_ground & below
    return int(np.sum(supported)) / total_above


# ---------------------------------------------------------------------------
# Symmetry Metrics
# ---------------------------------------------------------------------------

def _bilateral_symmetry_iou(grid: np.ndarray, axis: int) -> float:
    """
    Compute bilateral symmetry along the given axis as IoU of non-air voxels
    between the original and reflected grid.

    Args:
        grid: 3D voxel grid (X, Y, Z).
        axis: 0 for X-axis reflection, 2 for Z-axis reflection.

    Returns:
        IoU in [0, 1]. Returns 0 if no non-air blocks.
    """
    solid = grid != AIR
    reflected = np.flip(solid, axis=axis)

    intersection = int(np.sum(solid & reflected))
    union = int(np.sum(solid | reflected))

    if union == 0:
        return 0.0
    return intersection / union


def compute_bilateral_symmetry_x(grid: np.ndarray) -> float:
    """Bilateral symmetry along the X axis (left-right reflection)."""
    return _bilateral_symmetry_iou(grid, axis=0)


def compute_bilateral_symmetry_z(grid: np.ndarray) -> float:
    """Bilateral symmetry along the Z axis (front-back reflection)."""
    return _bilateral_symmetry_iou(grid, axis=2)


# ---------------------------------------------------------------------------
# Diversity Metrics
# ---------------------------------------------------------------------------

def compute_block_type_count(grid: np.ndarray) -> int:
    """Number of distinct non-air block types used."""
    unique = np.unique(grid)
    return int(np.sum(unique != AIR))


def compute_block_type_entropy(grid: np.ndarray) -> float:
    """
    Shannon entropy of the block type distribution, excluding air.
    Higher entropy means more diverse material usage.

    Returns:
        Entropy in nats. Returns 0.0 if fewer than 2 block types.
    """
    non_air = grid[grid != AIR]
    if len(non_air) == 0:
        return 0.0

    _, counts = np.unique(non_air, return_counts=True)
    if len(counts) <= 1:
        return 0.0

    probs = counts / counts.sum()
    # Shannon entropy: -sum(p * log(p))
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy)


def compute_spatial_block_clustering(grid: np.ndarray) -> float:
    """
    For each block type with >= 2 instances, compute the mean pairwise
    distance between blocks of that type, then compare against the expected
    mean distance under uniform random placement in the same volume.

    Returns:
        Average (actual_mean_dist / expected_mean_dist) across block types.
        Lower values (<1) indicate spatial clustering. Range ~[0, inf).
        Returns 0.0 if fewer than 2 block types have >= 2 instances.

    For computational efficiency with large counts, we subsample up to 200
    positions per block type for pairwise distance computation.
    """
    non_air_mask = grid != AIR
    if np.sum(non_air_mask) < 2:
        return 0.0

    # Expected mean distance in a 3D cube of side L under uniform placement:
    # E[d] ~ 0.6616 * L (well-known result for Manhattan-like 3D)
    # We use the actual average pairwise Euclidean distance for a uniform
    # distribution: E[d] ~ 0.6616 * L for unit cube, scale by grid dimensions.
    # For a WxHxD box: approximate E[d] = 0.6616 * (W^2 + H^2 + D^2)^0.5 / sqrt(3)
    # Simpler: estimate with a random sample
    L = float(GRID_SIZE)
    # Analytical approximation for mean Euclidean distance in [0,L]^3:
    # We'll just use L * 0.661 (normalized) which gives ~21.2 for L=32
    expected_mean_dist = L * 0.661

    unique_types = np.unique(grid[non_air_mask])
    MAX_SAMPLE = 200
    ratios = []

    for btype in unique_types:
        coords = np.argwhere(grid == btype)
        n = len(coords)
        if n < 2:
            continue

        # Subsample for efficiency
        if n > MAX_SAMPLE:
            rng = np.random.default_rng(seed=int(btype))
            idx = rng.choice(n, MAX_SAMPLE, replace=False)
            coords = coords[idx]
            n = MAX_SAMPLE

        # Compute mean pairwise distance
        # For up to 200 points: (200 choose 2) = 19900 pairs — fast enough
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (n, n, 3)
        dists = np.sqrt(np.sum(diff ** 2, axis=2))  # (n, n)
        # Upper triangle only (exclude diagonal)
        upper_mask = np.triu_indices(n, k=1)
        mean_dist = np.mean(dists[upper_mask])

        if expected_mean_dist > 0:
            ratios.append(mean_dist / expected_mean_dist)

    if len(ratios) == 0:
        return 0.0
    return float(np.mean(ratios))


# ---------------------------------------------------------------------------
# Core Evaluation
# ---------------------------------------------------------------------------

def evaluate_single_build(grid: np.ndarray, filename: str = "") -> StructuralMetrics:
    """Compute all structural metrics for a single 3D voxel grid."""
    metrics = StructuralMetrics(filename=filename)

    total = grid.size
    non_air = int(np.sum(grid != AIR))
    metrics.total_non_air = non_air
    metrics.total_air = total - non_air

    # Skip degenerate cases
    if non_air == 0:
        return metrics

    # Geometric
    metrics.enclosed_volume_ratio = compute_enclosed_volume_ratio(grid)
    metrics.surface_complexity = compute_surface_complexity(grid)
    metrics.height_utilization = compute_height_utilization(grid)
    metrics.footprint_density = compute_footprint_density(grid)

    # Structural coherence
    metrics.wall_continuity = compute_wall_continuity(grid)
    metrics.largest_connected_component_ratio = compute_largest_connected_component_ratio(grid)
    metrics.vertical_support_ratio = compute_vertical_support_ratio(grid)

    # Symmetry
    metrics.bilateral_symmetry_x = compute_bilateral_symmetry_x(grid)
    metrics.bilateral_symmetry_z = compute_bilateral_symmetry_z(grid)
    metrics.best_symmetry = max(metrics.bilateral_symmetry_x,
                                metrics.bilateral_symmetry_z)

    # Diversity
    metrics.block_type_count = compute_block_type_count(grid)
    metrics.block_type_entropy = compute_block_type_entropy(grid)
    metrics.spatial_block_clustering = compute_spatial_block_clustering(grid)

    return metrics


def _worker_evaluate(args: Tuple[str, str]) -> Optional[dict]:
    """
    Worker function for multiprocessing. Loads an .npz file and evaluates it.

    Args:
        args: (file_path, filename) tuple.

    Returns:
        Dict of metrics, or None on failure.
    """
    file_path, filename = args
    try:
        data = np.load(file_path)
        # Support common key conventions: 'grid', 'blocks', 'voxels', or first key
        if "grid" in data:
            grid = data["grid"]
        elif "blocks" in data:
            grid = data["blocks"]
        elif "voxels" in data:
            grid = data["voxels"]
        else:
            # Use the first array
            keys = list(data.keys())
            if not keys:
                return None
            grid = data[keys[0]]

        # Validate shape
        if grid.ndim != 3:
            print(f"  [WARN] {filename}: expected 3D array, got shape {grid.shape}")
            return None

        grid = grid.astype(np.int32)
        metrics = evaluate_single_build(grid, filename=filename)
        return asdict(metrics)

    except Exception as e:
        print(f"  [ERROR] {filename}: {e}")
        return None


# ---------------------------------------------------------------------------
# Aggregation & Comparison
# ---------------------------------------------------------------------------

def compute_summary_stats(metrics_list: List[dict], metric_names: List[str]) -> dict:
    """
    Compute mean, std, median, min, max for each metric across samples.

    Returns:
        Dict mapping metric_name -> {mean, std, median, min, max}.
    """
    summary = {}
    for name in metric_names:
        values = [m[name] for m in metrics_list if m is not None]
        if not values:
            summary[name] = {"mean": 0, "std": 0, "median": 0, "min": 0, "max": 0}
            continue
        arr = np.array(values, dtype=float)
        summary[name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    return summary


def format_comparison_table(
    gen_summary: dict,
    train_summary: dict,
    metric_names: List[str],
) -> str:
    """Format a side-by-side comparison table for stdout."""
    lines = []

    # Header
    header = (
        f"{'Metric':<38} "
        f"{'Gen Mean':>10} {'Gen Std':>10} "
        f"{'Train Mean':>10} {'Train Std':>10} "
        f"{'Delta':>10}"
    )
    sep = "=" * len(header)
    lines.append(sep)
    lines.append("  STRUCTURAL EVALUATION — Generated vs Training")
    lines.append(sep)
    lines.append(header)
    lines.append("-" * len(header))

    # Category labels
    categories = {
        "enclosed_volume_ratio": "GEOMETRIC",
        "wall_continuity": "STRUCTURAL COHERENCE",
        "bilateral_symmetry_x": "SYMMETRY",
        "block_type_count": "DIVERSITY",
    }

    for name in metric_names:
        if name in categories:
            lines.append(f"\n  [{categories[name]}]")

        g = gen_summary.get(name, {})
        t = train_summary.get(name, {})
        g_mean = g.get("mean", 0)
        g_std = g.get("std", 0)
        t_mean = t.get("mean", 0)
        t_std = t.get("std", 0)
        delta = g_mean - t_mean

        # Format based on type (count vs ratio)
        if name in ("block_type_count",):
            lines.append(
                f"  {name:<36} "
                f"{g_mean:>10.1f} {g_std:>10.1f} "
                f"{t_mean:>10.1f} {t_std:>10.1f} "
                f"{delta:>+10.1f}"
            )
        else:
            lines.append(
                f"  {name:<36} "
                f"{g_mean:>10.4f} {g_std:>10.4f} "
                f"{t_mean:>10.4f} {t_std:>10.4f} "
                f"{delta:>+10.4f}"
            )

    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# I/O Helpers
# ---------------------------------------------------------------------------

def collect_npz_files(directory: str, max_samples: int = 0) -> List[Tuple[str, str]]:
    """
    Collect .npz file paths from a directory.

    Returns:
        List of (full_path, filename) tuples, sorted by name.
    """
    dirpath = Path(directory)
    if not dirpath.exists():
        print(f"  [WARN] Directory not found: {directory}")
        return []

    files = sorted(dirpath.glob("*.npz"))
    if max_samples > 0:
        files = files[:max_samples]

    return [(str(f), f.name) for f in files]


def save_per_sample_csv(metrics_list: List[dict], output_path: str) -> None:
    """Save per-sample metrics to a CSV file."""
    if not metrics_list:
        return

    keys = list(metrics_list[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in metrics_list:
            writer.writerow(row)


def save_results_json(results: dict, output_path: str) -> None:
    """Save full results (summaries + per-sample) to JSON."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate_directory(
    npz_files: List[Tuple[str, str]],
    label: str,
    num_workers: int,
) -> List[dict]:
    """Evaluate all builds in a list of .npz files using multiprocessing."""
    n = len(npz_files)
    if n == 0:
        print(f"  No .npz files found for [{label}].")
        return []

    print(f"  [{label}] Evaluating {n} builds with {num_workers} workers...")
    t0 = time.time()

    if num_workers <= 1:
        results = [_worker_evaluate(args) for args in npz_files]
    else:
        with Pool(processes=num_workers) as pool:
            results = pool.map(_worker_evaluate, npz_files)

    # Filter failures
    results = [r for r in results if r is not None]
    elapsed = time.time() - t0
    print(f"  [{label}] Done. {len(results)}/{n} succeeded in {elapsed:.1f}s "
          f"({elapsed / max(n, 1) * 1000:.0f} ms/sample)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Structural evaluation metrics for 3D voxel building generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--generated_dir",
        type=str,
        default="outputs/generated/",
        help="Directory containing generated .npz files.",
    )
    parser.add_argument(
        "--training_dir",
        type=str,
        default="data/processed/builds/",
        help="Directory containing training .npz files for comparison.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/eval/",
        help="Directory for output files.",
    )
    parser.add_argument(
        "--max_training_samples",
        type=int,
        default=500,
        help="Maximum number of training samples to evaluate (0 = all).",
    )
    parser.add_argument(
        "--max_generated_samples",
        type=int,
        default=0,
        help="Maximum number of generated samples to evaluate (0 = all).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel workers (0 = auto-detect).",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root (parent of scripts/)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    gen_dir = Path(args.generated_dir)
    if not gen_dir.is_absolute():
        gen_dir = project_root / gen_dir

    train_dir = Path(args.training_dir)
    if not train_dir.is_absolute():
        train_dir = project_root / train_dir

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    num_workers = args.workers if args.workers > 0 else max(1, cpu_count() - 1)

    print("\n" + "=" * 72)
    print("  STRUCTURAL EVALUATION — 3D Voxel Building Generation")
    print("=" * 72)
    print(f"  Generated dir : {gen_dir}")
    print(f"  Training dir  : {train_dir}")
    print(f"  Output dir    : {output_dir}")
    print(f"  Workers       : {num_workers}")
    print()

    # ---- Collect files ----
    gen_files = collect_npz_files(str(gen_dir), max_samples=args.max_generated_samples)
    train_files = collect_npz_files(str(train_dir), max_samples=args.max_training_samples)

    if not gen_files and not train_files:
        print("  [ERROR] No .npz files found in either directory. Exiting.")
        sys.exit(1)

    # ---- Evaluate ----
    gen_metrics = evaluate_directory(gen_files, "GENERATED", num_workers)
    train_metrics = evaluate_directory(train_files, "TRAINING", num_workers)

    # ---- Compute summaries ----
    metric_names = StructuralMetrics().metric_names()

    gen_summary = compute_summary_stats(gen_metrics, metric_names) if gen_metrics else {}
    train_summary = compute_summary_stats(train_metrics, metric_names) if train_metrics else {}

    # ---- Output ----
    # 1. Comparison table to stdout
    if gen_summary and train_summary:
        table = format_comparison_table(gen_summary, train_summary, metric_names)
        print("\n" + table + "\n")
    elif gen_summary:
        print("\n  [INFO] Only generated metrics available (no training data).")
        table = format_comparison_table(gen_summary, gen_summary, metric_names)
        print(table + "\n")
    elif train_summary:
        print("\n  [INFO] Only training metrics available (no generated data).")
        table = format_comparison_table(train_summary, train_summary, metric_names)
        print(table + "\n")

    # 2. Per-sample CSV
    if gen_metrics:
        csv_path = str(output_dir / "generated_structural_metrics.csv")
        save_per_sample_csv(gen_metrics, csv_path)
        print(f"  Saved generated per-sample CSV: {csv_path}")

    if train_metrics:
        csv_path = str(output_dir / "training_structural_metrics.csv")
        save_per_sample_csv(train_metrics, csv_path)
        print(f"  Saved training per-sample CSV : {csv_path}")

    # 3. Full JSON results
    full_results = {
        "metadata": {
            "generated_dir": str(gen_dir),
            "training_dir": str(train_dir),
            "generated_count": len(gen_metrics),
            "training_count": len(train_metrics),
            "grid_size": GRID_SIZE,
            "metric_definitions": {
                "enclosed_volume_ratio":
                    "Interior air unreachable by exterior flood fill / total air. "
                    "Higher = more enclosed rooms.",
                "surface_complexity":
                    "Exposed surface area / minimum surface area for same volume. "
                    "Higher = more complex shape.",
                "height_utilization":
                    "Vertical span of structure / grid height. "
                    "Higher = taller buildings.",
                "footprint_density":
                    "Occupied XZ cells / XZ bounding box area. "
                    "Higher = denser footprint.",
                "wall_continuity":
                    "Mean fraction of horizontal neighbors that are also surface blocks. "
                    "Higher = smoother walls.",
                "largest_connected_component_ratio":
                    "Largest 6-connected component / total non-air blocks. "
                    "Higher = single coherent structure.",
                "vertical_support_ratio":
                    "Above-ground blocks with support below / total above-ground blocks. "
                    "Higher = fewer floating elements.",
                "bilateral_symmetry_x":
                    "IoU of non-air voxels with X-axis reflection. "
                    "Higher = more left-right symmetric.",
                "bilateral_symmetry_z":
                    "IoU of non-air voxels with Z-axis reflection. "
                    "Higher = more front-back symmetric.",
                "best_symmetry":
                    "max(bilateral_symmetry_x, bilateral_symmetry_z).",
                "block_type_count":
                    "Number of distinct non-air block types used.",
                "block_type_entropy":
                    "Shannon entropy of block type distribution (nats, excl. air). "
                    "Higher = more diverse materials.",
                "spatial_block_clustering":
                    "Mean pairwise distance per block type / expected random distance. "
                    "Lower (<1) = blocks of same type are spatially clustered.",
            },
        },
        "generated": {
            "summary": gen_summary,
            "per_sample": gen_metrics,
        },
        "training": {
            "summary": train_summary,
            "per_sample": train_metrics,
        },
    }

    json_path = str(output_dir / "structural_metrics.json")
    save_results_json(full_results, json_path)
    print(f"  Saved full results JSON       : {json_path}")

    # 4. Quick summary counts
    print()
    print(f"  Generated samples evaluated: {len(gen_metrics)}")
    print(f"  Training samples evaluated : {len(train_metrics)}")
    print()

    # 5. Key observations (auto-generated)
    if gen_summary and train_summary:
        print("  KEY OBSERVATIONS:")
        print("  " + "-" * 40)

        # Check a few important metrics
        checks = [
            ("enclosed_volume_ratio",
             "Enclosed volume",
             "Generated builds {cmp} enclosure than training ({gv:.3f} vs {tv:.3f})"),
            ("largest_connected_component_ratio",
             "Structural coherence",
             "Generated builds are {cmp} connected ({gv:.3f} vs {tv:.3f})"),
            ("vertical_support_ratio",
             "Physical plausibility",
             "Vertical support: generated {gv:.3f} vs training {tv:.3f} — {cmp} realistic"),
            ("best_symmetry",
             "Symmetry",
             "Best symmetry: generated {gv:.3f} vs training {tv:.3f}"),
            ("block_type_entropy",
             "Material diversity",
             "Block entropy: generated {gv:.3f} vs training {tv:.3f}"),
        ]

        for metric, label, template in checks:
            gv = gen_summary[metric]["mean"]
            tv = train_summary[metric]["mean"]
            delta = gv - tv

            if abs(delta) < 0.01:
                cmp = "similar"
            elif delta > 0:
                cmp = "more" if "less" not in template else "more"
            else:
                cmp = "less" if "less" not in template else "less"

            print(f"  - {template.format(cmp=cmp, gv=gv, tv=tv)}")

        print()

    print("=" * 72)
    print("  Evaluation complete.")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
