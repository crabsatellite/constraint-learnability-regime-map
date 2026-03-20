"""
Dense voxel dataset for VQ-VAE training.

Loads processed MC builds and pads them to 32x32x32 grids.
Supports data augmentation: 4 rotations x 2 flips = 8x augmentation.
Uses remapped vocab (top-512 block types + air = 513 total).
"""

import csv
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class DenseVoxelDataset(Dataset):
    """Loads MC builds as dense 32x32x32 voxel grids with remapped tokens."""

    def __init__(self, processed_dir, max_dim=32, min_blocks=20,
                 sources=None, augment=True):
        self.processed_dir = Path(processed_dir)
        self.max_dim = max_dim
        self.augment = augment

        # Load token remapping (old_token -> new_token, 0-512)
        remap_path = self.processed_dir / "token_remap.json"
        with open(remap_path, "r") as f:
            raw_remap = json.load(f)
        # Build numpy lookup table for fast remapping
        max_old = max(int(k) for k in raw_remap.keys())
        self._remap_table = np.zeros(max_old + 1, dtype=np.int64)
        for old_str, new_id in raw_remap.items():
            self._remap_table[int(old_str)] = new_id
        self.vocab_size = max(raw_remap.values()) + 1
        print(f"Vocab remapped: {len(raw_remap)} entries -> {self.vocab_size} classes")

        # Load manifest
        with open(self.processed_dir / "manifest.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            entries = list(reader)

        # Filter
        self.entries = []
        for m in entries:
            if sources and m["source"] not in sources:
                continue
            if int(m["non_air_blocks"]) < min_blocks:
                continue
            sx, sy, sz = int(m["shape_x"]), int(m["shape_y"]), int(m["shape_z"])
            if max(sx, sy, sz) > max_dim:
                continue
            self.entries.append(m)

        # Cache for loaded arrays
        self._cache = {}

    def __len__(self):
        return len(self.entries)

    def _load(self, idx):
        if idx in self._cache:
            return self._cache[idx]

        entry = self.entries[idx]
        path = self.processed_dir / entry["path"]
        data = np.load(path)
        voxels = data["voxels"]  # (X, Y, Z) uint16, variable shape

        # Pad to max_dim³, centered in XZ, grounded in Y
        sx, sy, sz = voxels.shape
        padded = np.zeros((self.max_dim, self.max_dim, self.max_dim), dtype=np.int64)
        ox = (self.max_dim - sx) // 2
        oy = 0
        oz = (self.max_dim - sz) // 2
        padded[ox:ox+sx, oy:oy+sy, oz:oz+sz] = voxels.astype(np.int64)

        # Remap tokens: clamp out-of-range to 0, then apply lookup table
        padded[padded >= len(self._remap_table)] = 0
        padded = self._remap_table[padded]

        self._cache[idx] = padded
        return padded

    def __getitem__(self, idx):
        voxels = self._load(idx).copy()

        if self.augment:
            k = random.randint(0, 3)
            if k > 0:
                voxels = np.rot90(voxels, k=k, axes=(0, 2)).copy()
            if random.random() > 0.5:
                voxels = np.flip(voxels, axis=0).copy()

        return torch.tensor(voxels, dtype=torch.long)
