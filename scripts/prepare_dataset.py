"""
Unified dataset preparation for MC building generation.

Processes text2mc, 3D-Craft, and rom1504 into a single unified format:
- Each build is saved as a numpy .npz file
- Contains: voxels (uint16 3D array), metadata dict
- Unified block vocabulary (modern namespace)
- Filtered for quality (no empty builds, no terrain-only)
"""

import os
import sys
import json
import struct
import gzip
import io
import numpy as np
import h5py
from pathlib import Path
from collections import Counter
from legacy_block_map import legacy_to_modern

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Unified vocabulary ---

def build_unified_vocab():
    """Build a unified block vocabulary from text2mc's tok2block.json
    plus any additional blocks from 3D-Craft legacy mapping."""

    # Start with text2mc vocabulary (modern namespace, most complete)
    tok2block_path = RAW_DIR / "text2mc" / "tok2block.json"
    with open(tok2block_path, "r") as f:
        tok2block = json.load(f)

    # Build modern name -> unified token mapping
    # Token 0 = air (reserved)
    block_to_token = {"minecraft:air": 0}
    token_to_block = {0: "minecraft:air"}
    next_token = 1

    # First, add all text2mc blocks (preserving their detailed states)
    for old_tok, block_name in sorted(tok2block.items(), key=lambda x: int(x[0])):
        if block_name == "minecraft:air":
            continue
        if block_name not in block_to_token:
            block_to_token[block_name] = next_token
            token_to_block[next_token] = block_name
            next_token += 1

    # Then add any 3D-Craft blocks not already in vocab (strip states for legacy)
    from legacy_block_map import LEGACY_TO_MODERN
    for (bid, meta), modern_name in LEGACY_TO_MODERN.items():
        if modern_name not in block_to_token:
            block_to_token[modern_name] = next_token
            token_to_block[next_token] = modern_name
            next_token += 1

    print(f"Unified vocabulary: {next_token} tokens (0=air)")
    return block_to_token, token_to_block


# --- text2mc processing ---

# text2mc terrain tokens (used for filtering)
TEXT2MC_TERRAIN_NAMES = {
    "minecraft:air", "minecraft:cave_air", "minecraft:void_air",
    "minecraft:stone", "minecraft:dirt", "minecraft:grass_block",
    "minecraft:coal_ore", "minecraft:iron_ore", "minecraft:gold_ore",
    "minecraft:diamond_ore", "minecraft:copper_ore", "minecraft:lapis_ore",
    "minecraft:emerald_ore", "minecraft:redstone_ore",
    "minecraft:granite", "minecraft:diorite", "minecraft:andesite",
    "minecraft:deepslate", "minecraft:tuff", "minecraft:gravel",
    "minecraft:sand", "minecraft:bedrock", "minecraft:snow_layer",
    "minecraft:grass", "minecraft:tall_grass",
}


def process_text2mc(block_to_token, max_dim=64):
    """Process text2mc HDF5 files into unified format."""
    print("\n=== Processing text2mc ===")

    # Load text2mc's own token mapping
    tok2block_path = RAW_DIR / "text2mc" / "tok2block.json"
    with open(tok2block_path, "r") as f:
        tok2block = json.load(f)

    # Build terrain token set for this dataset
    terrain_tokens = set()
    for old_tok, block_name in tok2block.items():
        base_name = block_name.split("[")[0]
        if base_name in TEXT2MC_TERRAIN_NAMES:
            terrain_tokens.add(int(old_tok))

    # Build old_token -> unified_token mapping
    old_to_unified = {}
    for old_tok, block_name in tok2block.items():
        if block_name in block_to_token:
            old_to_unified[int(old_tok)] = block_to_token[block_name]
        else:
            # Should not happen if vocab was built correctly
            old_to_unified[int(old_tok)] = 0  # map to air

    import zipfile
    archive_path = RAW_DIR / "text2mc" / "archive.zip"

    # Check if archive exists or if files are already extracted
    h5_dir = RAW_DIR / "text2mc" / "processed_builds" / "processed_builds"
    if h5_dir.exists():
        h5_files = list(h5_dir.glob("*.h5"))
        use_zip = False
    elif archive_path.exists():
        use_zip = True
        z = zipfile.ZipFile(archive_path, "r")
        h5_files = [n for n in z.namelist() if n.endswith(".h5")]
    else:
        print("WARNING: text2mc data not found (no archive.zip or extracted files)")
        return []

    results = []
    skipped_empty = 0
    skipped_terrain = 0
    skipped_large = 0

    total = len(h5_files)
    for i, fname in enumerate(h5_files):
        if i % 2000 == 0:
            print(f"  {i}/{total}...", file=sys.stderr)

        try:
            if use_zip:
                data_bytes = z.read(fname)
                f = h5py.File(io.BytesIO(data_bytes), "r")
            else:
                f = h5py.File(str(fname), "r")

            key = list(f.keys())[0]
            arr = f[key][()]
            f.close()

            # Check empty
            air_token = 102  # text2mc air token
            non_air_mask = arr != air_token
            non_air_count = np.sum(non_air_mask)
            if non_air_count == 0:
                skipped_empty += 1
                continue

            # Check terrain-dominant (>95%)
            terrain_count = sum(int(np.sum(arr == t)) for t in terrain_tokens)
            if terrain_count / arr.size > 0.95:
                skipped_terrain += 1
                continue

            # Check dimensions
            if max(arr.shape) > max_dim:
                skipped_large += 1
                continue

            # Remap tokens to unified vocabulary
            unified = np.zeros_like(arr, dtype=np.uint16)
            for old_tok in np.unique(arr):
                mask = arr == old_tok
                unified[mask] = old_to_unified.get(int(old_tok), 0)

            # Extract build name
            build_name = fname.split("/")[-1] if use_zip else fname.name
            build_name = build_name.replace(".h5", "")

            results.append({
                "name": f"text2mc_{build_name}",
                "voxels": unified,
                "source": "text2mc",
                "non_air_blocks": int(np.sum(unified != 0)),
                "shape": unified.shape,
            })

        except Exception as e:
            continue

    if use_zip:
        z.close()

    print(f"  text2mc: {len(results)} good builds")
    print(f"  Skipped: {skipped_empty} empty, {skipped_terrain} terrain, {skipped_large} too large")
    return results


# --- 3D-Craft processing ---

def process_3dcraft(block_to_token, min_blocks=50):
    """Process 3D-Craft houses into unified format."""
    print("\n=== Processing 3D-Craft ===")

    base = RAW_DIR / "3d-craft" / "houses"
    if not base.exists():
        print("WARNING: 3D-Craft data not found")
        return []

    results = []
    skipped_small = 0

    for h in sorted(os.listdir(base)):
        schem_path = base / h / "schematic.npy"
        if not schem_path.exists():
            continue

        arr = np.load(schem_path)  # (y, z, x, 2) with [id, meta]
        block_ids = arr[..., 0]
        block_metas = arr[..., 1]

        non_air = int(np.sum(block_ids != 0))
        if non_air < min_blocks:
            skipped_small += 1
            continue

        # Convert to unified tokens
        # arr shape is (y, z, x) after dropping last dim
        unified = np.zeros(block_ids.shape, dtype=np.uint16)
        for bid_val in np.unique(block_ids):
            if bid_val == 0:
                continue
            bid_mask = block_ids == bid_val
            for meta_val in np.unique(block_metas[bid_mask]):
                combo_mask = bid_mask & (block_metas == meta_val)
                modern_name = legacy_to_modern(int(bid_val), int(meta_val))
                token = block_to_token.get(modern_name, 0)
                unified[combo_mask] = token

        # Reorder from (y, z, x) to (x, y, z) for consistency
        unified = np.transpose(unified, (2, 0, 1))  # (x, y, z)

        results.append({
            "name": f"3dcraft_{h}",
            "voxels": unified,
            "source": "3d-craft",
            "non_air_blocks": int(np.sum(unified != 0)),
            "shape": unified.shape,
        })

    print(f"  3D-Craft: {len(results)} builds (skipped {skipped_small} too small)")
    return results


# --- rom1504 processing ---

def read_tfrecord(f):
    buf = f.read(8)
    if not buf or len(buf) < 8:
        return None
    length = struct.unpack("<Q", buf)[0]
    f.read(4)
    data = f.read(length)
    f.read(4)
    return data


def process_rom1504(block_to_token, min_blocks=20):
    """Process rom1504 good_small.tfrecord.gz into unified format.

    Note: rom1504 stores raw schematic bytes as int16.
    The block encoding is legacy format packed as (id | meta<<8) in some entries
    and raw schematic bytes in others. We treat each unique int16 value as a
    distinct block type and map what we can.
    """
    print("\n=== Processing rom1504 ===")

    filepath = RAW_DIR / "rom1504" / "data" / "good_small.tfrecord.gz"
    if not filepath.exists():
        print("WARNING: rom1504 data not found")
        return []

    # Load metadata for tags
    meta_path = RAW_DIR / "rom1504" / "data" / "schematicsWithFinalUrl.json"
    url_to_meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_list = json.load(f)
        url_to_meta = {d["url"]: d for d in meta_list}

    results = []
    skipped_small = 0
    count = 0

    with gzip.open(filepath, "rb") as f:
        while True:
            record = read_tfrecord(f)
            if record is None:
                break
            count += 1

            # Extract URL
            url_idx = record.find(b"url")
            url = ""
            if url_idx >= 0:
                # Simple extraction: find http after "url"
                http_idx = record.find(b"http", url_idx)
                if http_idx >= 0:
                    end = http_idx
                    while end < len(record) and record[end] >= 32 and record[end] < 127:
                        end += 1
                    url = record[http_idx:end].decode("ascii", errors="ignore")

            # Extract schematic data
            idx = record.find(b"schematicData")
            if idx < 0:
                continue

            remaining = record[idx:]
            arr = None
            for offset in range(13, min(len(remaining), 100)):
                if len(remaining) >= offset + 65536:
                    try:
                        chunk = remaining[offset : offset + 65536]
                        arr = np.frombuffer(chunk, dtype=np.int16).reshape(32, 32, 32).copy()
                        non_air = np.sum(arr != 0)
                        if 0 < non_air < 32768:
                            break
                        arr = None
                    except:
                        arr = None

            if arr is None:
                continue

            non_air = int(np.sum(arr != 0))
            if non_air < min_blocks:
                skipped_small += 1
                continue

            # For rom1504, we keep the raw int16 values as tokens for now.
            # The model can learn this vocabulary independently.
            # We offset all values by 5000 to avoid collision with text2mc/3dcraft tokens.
            unified = np.zeros_like(arr, dtype=np.uint16)
            for val in np.unique(arr):
                if val == 0:
                    continue
                mask = arr == val
                # Map to offset range to avoid collision
                unified[mask] = 5000 + (int(val) & 0xFFFF)

            # Get tags from metadata
            meta = url_to_meta.get(url, {})
            tags = meta.get("tags", [])
            title = meta.get("title", "")

            results.append({
                "name": f"rom1504_{count}",
                "voxels": unified,
                "source": "rom1504",
                "non_air_blocks": non_air,
                "shape": unified.shape,
                "tags": tags,
                "title": title,
                "url": url,
            })

    print(f"  rom1504: {len(results)} builds (skipped {skipped_small} too small)")
    return results


# --- Save dataset ---

def save_dataset(all_builds, token_to_block):
    """Save all builds as individual .npz files + manifest CSV."""
    print(f"\n=== Saving {len(all_builds)} builds ===")

    # Create output directories by source
    for source in ["text2mc", "3d-craft", "rom1504"]:
        (OUT_DIR / source).mkdir(parents=True, exist_ok=True)

    # Save vocab
    vocab_path = OUT_DIR / "token_to_block.json"
    with open(vocab_path, "w") as f:
        json.dump({str(k): v for k, v in token_to_block.items()}, f, indent=2)
    print(f"  Saved vocabulary ({len(token_to_block)} tokens) to {vocab_path}")

    # Save each build
    manifest = []
    for build in all_builds:
        source = build["source"]
        name = build["name"]
        outpath = OUT_DIR / source / f"{name}.npz"

        np.savez_compressed(
            outpath,
            voxels=build["voxels"],
        )

        entry = {
            "name": name,
            "source": source,
            "path": str(outpath.relative_to(OUT_DIR)),
            "non_air_blocks": build["non_air_blocks"],
            "shape_x": build["shape"][0],
            "shape_y": build["shape"][1],
            "shape_z": build["shape"][2],
        }
        if "tags" in build:
            entry["tags"] = "|".join(build["tags"])
        if "title" in build:
            entry["title"] = build["title"]
        manifest.append(entry)

    # Save manifest
    import csv
    manifest_path = OUT_DIR / "manifest.csv"
    if manifest:
        keys = list(manifest[0].keys())
        # Add optional fields
        all_keys = set()
        for m in manifest:
            all_keys.update(m.keys())
        keys = sorted(all_keys)

        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(manifest)
        print(f"  Saved manifest ({len(manifest)} entries) to {manifest_path}")

    # Print summary by source
    source_counts = Counter(b["source"] for b in all_builds)
    print("\n=== Dataset Summary ===")
    for source, count in source_counts.most_common():
        builds = [b for b in all_builds if b["source"] == source]
        blocks = [b["non_air_blocks"] for b in builds]
        print(f"  {source}: {count} builds, "
              f"blocks: min={min(blocks)}, median={sorted(blocks)[len(blocks)//2]}, max={max(blocks)}")

    # Dimension analysis
    for max_d in [32, 64, 128]:
        fits = sum(1 for b in all_builds if max(b["shape"]) <= max_d)
        print(f"  Fits in {max_d}x{max_d}x{max_d}: {fits}")


def main():
    print("Building unified vocabulary...")
    block_to_token, token_to_block = build_unified_vocab()

    all_builds = []

    # Process each dataset
    text2mc_builds = process_text2mc(block_to_token, max_dim=128)
    all_builds.extend(text2mc_builds)

    craft_builds = process_3dcraft(block_to_token, min_blocks=50)
    all_builds.extend(craft_builds)

    rom1504_builds = process_rom1504(block_to_token, min_blocks=20)
    all_builds.extend(rom1504_builds)

    # Save everything
    save_dataset(all_builds, token_to_block)

    print(f"\n{'='*50}")
    print(f"TOTAL: {len(all_builds)} builds ready for training")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
