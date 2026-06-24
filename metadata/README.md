# Metadata

This directory contains small reproducibility metadata for the voxel
experiments.

## `remapped_vocab_513.json`

Project-specific voxel vocabulary for the 513-class voxel space used by
preprocessing and by the VQ-VAE input/output heads:

- token `0`: air
- tokens `1`-`512`: the most frequent non-air source tokens in the
  filtered VQ-VAE training set, remapped to a compact ID range

This is not an official Minecraft block vocabulary. It is the vocabulary used
by this paper's preprocessing pipeline after merging the three data sources,
applying the VQ-VAE training filters, and remapping source block/material
identifiers to a compact 513-class voxel space.

This JSON is the public, human-readable export of that remap, with block names
filled in where source tokens resolve to modern Minecraft names.

Some entries resolve to modern Minecraft block names. Entries from the rom1504
source may remain raw int16 schematic tokens; the model treats each distinct
raw value as a distinct material token, as described in `scripts/prepare_dataset.py`.
