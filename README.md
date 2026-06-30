# Which Structural Constraints Are Learnable?

Companion repository for *Which Structural Constraints Are Learnable? A Regime
Map for a Minecraft Voxel Generator*.

This project is a diagnostic case study for learned PCG control: given a fixed
neural generator and a set of structural constraints, which properties actually
respond to conditioning? The released VQ-VAE + autoregressive Transformer is the
experimental pipeline used to build and audit the map. It is not presented as a
state-of-the-art Minecraft generator.

## Key Results

Pipeline-specific results for the archived generator pipeline:

| Regime | Properties | Ctrl% Range | Bottleneck |
|--------|------------|-------------|------------|
| Controllable | 9 / 14 | >100% | None for the measured response; standard CFG works |
| Approachable | 4 / 14 | 20--100% | Frequency floor (more data, stronger guidance, or fine-tuning may help) |
| Unresponsive | 1 / 14 | <20% | Representation ceiling or insufficient learnable variation |

`Ctrl%` is a relative shift of generated-sample means, not a guarantee that
every sample satisfies a hard constraint. The composite predictor, effective
signal x min(CV, 1), correlates with controllability at **Spearman rho = 0.879**
(permutation p = 0.002, n = 10 emergent properties). Neither signal alone
(0.624) nor CV alone (0.636) reaches significance under the same permutation
test; their product does. This is an association from one pipeline, not a
validated general predictor.

## What This Repository Is

- **Regime-map evidence package**: saved measurements and statistics for a
  three-regime constraint-learnability map across 14 structural properties,
  plus scripts that rebuild the analysis from those results.
- **Diagnostic protocol implementation**: train a fixed generator, condition it
  on structural tokens, generate samples, measure structural properties, and
  compare conditional distributions against unconditioned generation.
- **One Minecraft voxel case study**: 10,310 filtered builds, 32^3 voxel grids,
  and a released 513-token voxel vocabulary.
- **Fixed generator under test**: a VQ-VAE (32^3 -> 8^3 latent, 2048 codebook,
  ~40M params) plus autoregressive Transformer (512d, 12 layers, 8 heads,
  ~40M params) with classifier-free guidance.
- **Bottleneck and predictor analyses**: CFG sensitivity (s=0,2,4) is used to
  distinguish frequency floors from representation ceilings; the composite
  predictor is reported as a hypothesis for future validation.
- **[Interactive explainer](https://crabsatellite.github.io/constraint-learnability-regime-map/)**: bilingual (EN/ZH) step-by-step walkthrough of the paper for non-specialist audiences

## What This Is Not

- Not a claim that this generator is the best or most realistic Minecraft
  building generator.
- Not a universal table of which Minecraft constraints are always learnable.
  The map is tied to this dataset, architecture, conditioning scheme, and
  evaluation protocol.
- Not a replacement for hard constraint satisfaction methods. The map identifies
  where learned conditioning responds and where explicit constraints or
  architectural changes may be needed.

## Repository Structure

```text
constraint-learnability/
|-- index.html                             # Interactive paper explainer (bilingual EN/ZH)
|-- generate_scatter.py                    # Rebuild scatter_predictor from saved results
|-- figures/
|   |-- scatter_predictor.png              # Composite score vs controllability
|   |-- scatter_predictor.pdf              # Paper-ready vector version
|   |-- scatter_predictor_data.json        # Data used to draw scatter_predictor
|   |-- cfg_sensitivity.png                # CFG sensitivity visualization
|   |-- cfg_sensitivity.pdf                # Paper-ready vector version
|   |-- condition_comparison.png           # Condition comparison visualization
|   `-- condition_comparison.pdf           # Paper-ready vector version
|-- metadata/
|   |-- README.md                          # Notes on released metadata
|   `-- remapped_vocab_513.json            # 513-class voxel vocabulary
|-- models/
|   |-- vqvae.py                           # VQ-VAE (3D conv encoder-decoder, 2048 codebook)
|   `-- ar_transformer.py                  # AR Transformer with structural conditioning + CFG
|-- scripts/
|   |-- prepare_dataset.py                 # Unified dataset preparation (text2mc, 3D-Craft, rom1504)
|   |-- dataset_dense.py                   # Dense latent code dataset for AR training
|   |-- train_vqvae.py                     # VQ-VAE training (100K steps)
|   |-- extract_structural_features.py     # Structural feature extraction for conditioning
|   |-- train_ar_conditioned.py            # Conditioned AR training (80K steps, CFG)
|   |-- test_regime_map.py                 # Full regime map evaluation (5 seeds x 8 samples)
|   |-- test_cfg_sensitivity.py            # CFG sensitivity sweep (s=0,2,4)
|   |-- regime_predictor.py                # Predictor statistics and association analysis
|   |-- bootstrap_regime_map.py            # Bootstrap 95% CI computation
|   |-- supplementary_analysis.py          # Permutation tests and robustness checks
|   |-- generate.py                        # Generate builds from trained models
|   |-- visualize_structures.py            # Visualization utilities
|   `-- legacy_block_map.py                # Minecraft legacy block ID mapping
|-- outputs/
|   |-- regime_map_results.json            # Regime map samples and statistics
|   |-- cfg_sensitivity_results.json       # CFG sensitivity data
|   |-- regime_predictor_results.json      # Predictor feature table and statistics
|   |-- bootstrap_results.json             # Bootstrap 95% CI analysis
|   `-- supplementary_results.json         # Permutation and per-seed diagnostics
`-- requirements.txt
```

## Quick Start

### Option A: Reproduce analysis from saved results

All saved experimental results are included in `outputs/`. The
`generate_scatter.py` script only rebuilds the predictor scatter plot from
saved JSON results; it does not generate Minecraft structures.

```bash
pip install -r requirements.txt
python scripts/regime_predictor.py
python scripts/bootstrap_regime_map.py
python scripts/supplementary_analysis.py
python generate_scatter.py
```

### Option B: Train from scratch

The training script defaults match the v1.0.1 paper configuration.

```bash
pip install -r requirements.txt

# 1. Prepare dataset (requires raw data in data/raw/)
python scripts/prepare_dataset.py

# 2. Train VQ-VAE (100K steps)
python scripts/train_vqvae.py

# 3. Extract structural features for conditioning
python scripts/extract_structural_features.py

# 4. Train conditioned AR transformer (80K steps)
python scripts/train_ar_conditioned.py

# 5. Run regime map evaluation
python scripts/test_regime_map.py

# 6. Run CFG sensitivity analysis
python scripts/test_cfg_sensitivity.py

# 7. Fit predictor and supplementary analyses
python scripts/regime_predictor.py
python scripts/bootstrap_regime_map.py
python scripts/supplementary_analysis.py
```

Note on exact checkpoint provenance: the released conditioned AR checkpoint was
initialized from a compatible unconditioned AR checkpoint before conditioned
training. The command above trains the same conditioned architecture from
scratch unless `--resume` is supplied; use the released checkpoint assets to
reproduce the paper's generated outputs exactly.

## Data Availability

This repository does not redistribute the raw or processed Minecraft structure
training data. The experiments combine third-party datasets, and some upstream
sources do not provide clear redistribution licenses for repackaged data. To
train from scratch, obtain the upstream data separately and place it under
`data/raw/` before running `scripts/prepare_dataset.py`.

Expected raw-data layout:

```text
data/raw/
|-- text2mc/
|   |-- tok2block.json                    # required vocabulary source
|   `-- archive.zip                       # or processed_builds/processed_builds/*.h5
|-- 3d-craft/
|   `-- houses/*/schematic.npy
`-- rom1504/
    `-- data/good_small.tfrecord.gz
```

At least one structure source must be present. The Text2MC `tok2block.json`
file is required to build the unified vocabulary used by the preprocessing
script.

Trained checkpoints are not committed to Git because the final VQ-VAE and AR
Transformer checkpoints are large. They are attached to the
[v1.0.1 GitHub release](https://github.com/crabsatellite/constraint-learnability-regime-map/releases/tag/v1.0.1):

- `vqvae_step100000.pt` (477,353,957 bytes; SHA256 `c9caf9e76a1f1ef8512897cd2aafaa6099d4aa1e7a514cb5203707028e77e019`)
- `ar_cond_step80000.pt` (486,868,067 bytes; SHA256 `919b0c1190f940c98cdcd0c58e52c86b3c66702ae8d35b3b7926b2fb8aaec1e7`)
- `SHA256SUMS.txt`

The same assets are also included in the Zenodo all-versions record cited below.

The saved JSON outputs needed to reproduce the paper's tables and figures are
included in `outputs/`.

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- GPU recommended for training and fresh generation runs
- See [requirements.txt](requirements.txt) for dependencies

## Paper

*Which Structural Constraints Are Learnable? A Regime Map for a Minecraft Voxel Generator*

Alex Chengyu Li, 2026

Accepted at Foundations of Digital Games (FDG '26).

- Paper DOI: [10.1145/3815598.3815669](https://doi.org/10.1145/3815598.3815669)
- ACM booktitle: Foundations of Digital Games (FDG '26), August 10--13, 2026, Copenhagen, Denmark
- ACM ISBN: 979-8-4007-2495-4/2026/08
- Archived repository DOI: [10.5281/zenodo.20821894](https://doi.org/10.5281/zenodo.20821894)

## Citation

```bibtex
@inproceedings{li2026learnable,
  author = {Li, Alex Chengyu},
  title = {Which Structural Constraints Are Learnable? A Regime Map for a Minecraft Voxel Generator},
  booktitle = {Foundations of Digital Games (FDG '26), August 10--13, 2026, Copenhagen, Denmark},
  year = {2026},
  isbn = {979-8-4007-2495-4/2026/08},
  publisher = {Association for Computing Machinery},
  doi = {10.1145/3815598.3815669}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- 3D-Craft dataset by Chen et al. (CVPR 2019)
- rom1504 Minecraft schematics dataset
- VQ-VAE architecture based on van den Oord et al. (NeurIPS 2017)
