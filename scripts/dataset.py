"""
Sparse voxel dataset for MC building generation.

Each build is converted to a sparse sequence of (x, y, z, block_token) tuples,
sorted by Y (bottom-up), then Z, then X. Air voxels are excluded.
"""

import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class MCBuildDataset(Dataset):
    """Loads processed MC builds as sparse token sequences."""

    def __init__(self, processed_dir, max_seq_len=1024, max_dim=32,
                 sources=None, min_blocks=20, pad_token=0):
        self.processed_dir = Path(processed_dir)
        self.max_seq_len = max_seq_len
        self.max_dim = max_dim
        self.pad_token = pad_token

        # Load manifest
        manifest_path = self.processed_dir / "manifest.csv"
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.entries = list(reader)

        # Filter by source, size, and block count
        filtered = []
        for m in self.entries:
            if sources and m["source"] not in sources:
                continue
            if int(m["non_air_blocks"]) < min_blocks:
                continue
            max_s = max(int(m["shape_x"]), int(m["shape_y"]), int(m["shape_z"]))
            if max_s > max_dim:
                continue
            filtered.append(m)

        self.entries = filtered

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        path = self.processed_dir / entry["path"]

        # Load voxel array
        data = np.load(path)
        voxels = data["voxels"]  # (X, Y, Z) uint16

        # Convert to sparse sequence: list of (x, y, z, block_token)
        # Sort by Y (height), then Z, then X for natural building order
        coords = np.argwhere(voxels != 0)  # (N, 3) where columns are x, y, z
        if len(coords) == 0:
            # Empty build - return padded sequence
            seq = np.zeros((self.max_seq_len, 4), dtype=np.int64)
            mask = np.zeros(self.max_seq_len, dtype=np.bool_)
            return torch.tensor(seq), torch.tensor(mask)

        tokens = voxels[coords[:, 0], coords[:, 1], coords[:, 2]]

        # Sort by y, z, x (building order: bottom to top)
        sort_idx = np.lexsort((coords[:, 0], coords[:, 2], coords[:, 1]))
        coords = coords[sort_idx]
        tokens = tokens[sort_idx]

        # Build sequence: each element is (x, y, z, block_token)
        seq = np.stack([coords[:, 0], coords[:, 1], coords[:, 2], tokens], axis=1)

        # Truncate to max_seq_len
        if len(seq) > self.max_seq_len:
            seq = seq[:self.max_seq_len]

        n = len(seq)
        mask = np.ones(self.max_seq_len, dtype=np.bool_)
        mask[n:] = False

        # Pad to max_seq_len
        padded = np.zeros((self.max_seq_len, 4), dtype=np.int64)
        padded[:n] = seq

        return torch.tensor(padded, dtype=torch.long), torch.tensor(mask)


def get_tag_labels(processed_dir):
    """Extract unique tags for conditional generation."""
    manifest_path = Path(processed_dir) / "manifest.csv"
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        entries = list(reader)

    tag_set = set()
    for m in entries:
        tags = m.get("tags", "")
        if tags:
            for t in tags.split("|"):
                t = t.strip()
                if t:
                    tag_set.add(t)

    return sorted(tag_set)


if __name__ == "__main__":
    ds = MCBuildDataset(
        str(Path(__file__).parent.parent / "data" / "processed"),
        max_seq_len=1024,
        max_dim=32,
    )
    print(f"Dataset: {len(ds)} builds")

    # Test a few samples
    for i in range(min(5, len(ds))):
        seq, mask = ds[i]
        n_blocks = mask.sum().item()
        if n_blocks > 0:
            max_x = seq[:n_blocks, 0].max().item()
            max_y = seq[:n_blocks, 1].max().item()
            max_z = seq[:n_blocks, 2].max().item()
            n_unique = len(seq[:n_blocks, 3].unique())
            print(f"  [{i}] blocks={n_blocks}, dims=({max_x+1},{max_y+1},{max_z+1}), "
                  f"unique_tokens={n_unique}, "
                  f"entry={ds.entries[i]['name']}")
