# Which Structural Constraints Are Learnable?

Code, saved experimental results, and [interactive explainer](https://crabsatellite.github.io/constraint-learnability-regime-map/) for *Which Structural Constraints Are Learnable? A Regime Map for a Minecraft Voxel Generator*.

## Key Results

| Regime | Properties | Ctrl% Range | Bottleneck |
|--------|------------|-------------|------------|
| Controllable | 9 / 14 | >100% | None for the measured response; standard CFG works |
| Approachable | 4 / 14 | 20--100% | Frequency floor (more data, stronger guidance, or fine-tuning may help) |
| Unresponsive | 1 / 14 | <20% | Representation ceiling or insufficient learnable variation |

Composite predictor: effective signal x min(CV, 1) correlates with controllability at **Spearman rho = 0.879** (permutation p = 0.002, n = 10 emergent properties). Neither signal alone (0.624) nor CV alone (0.636) reaches significance under the same permutation test; their product does. This is an association from one pipeline, not a validated general predictor.

## What This Is

- **Three-regime learnability map** across 14 structural properties of Minecraft buildings (10,310 filtered builds, 32^3 voxel grids, 513 remapped block tokens)
- **VQ-VAE** (32^3 -> 8^3 latent, 2048 codebook, ~40M params) + **AR Transformer** (512d, 12 layers, 8 heads, ~40M params) with classifier-free guidance
- **Dual bottleneck analysis**: CFG sensitivity (s=0,2,4) helps diagnose frequency floors versus representation ceilings
- **Predictor analysis**: composite score (signal x min(CV, 1)) is tested against generated-sample responsiveness
- **[Interactive explainer](https://crabsatellite.github.io/constraint-learnability-regime-map/)**: bilingual (EN/ZH) step-by-step walkthrough of the paper for non-specialist audiences

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
|   |-- dataset.py                         # PyTorch dataset for VQ-VAE training
|   |-- dataset_dense.py                   # Dense latent code dataset for AR training
|   |-- train_vqvae.py                     # VQ-VAE training (100K steps)
|   |-- train_ar.py                        # Unconditional AR training
|   |-- train_ar_conditioned.py            # Conditioned AR training (80K steps, CFG)
|   |-- extract_structural_features.py     # Extract structural conditioning features
|   |-- test_regime_map.py                 # Full regime map evaluation (5 seeds x 8 samples)
|   |-- test_cfg_sensitivity.py            # CFG sensitivity sweep (s=0,2,4)
|   |-- test_conditioning.py               # Per-condition generation tests
|   |-- test_composability_ood.py          # Composability and OOD experiments
|   |-- regime_predictor.py                # Predictor statistics and association analysis
|   |-- bootstrap_regime_map.py            # Bootstrap 95% CI computation
|   |-- supplementary_analysis.py          # Permutation tests and robustness checks
|   |-- generate.py                        # Generate builds from trained models
|   |-- generate_conditioned.py            # Conditioned generation
|   |-- augment_symmetry.py                # Symmetry data augmentation
|   |-- run_symmetry_experiment.py         # Symmetry augmentation experiments
|   |-- visualize_structures.py            # Visualization utilities
|   `-- legacy_block_map.py                # Minecraft legacy block ID mapping
|-- outputs/
|   |-- regime_map_results.json            # Regime map samples and statistics
|   |-- cfg_sensitivity_results.json       # CFG sensitivity data
|   |-- regime_predictor_results.json      # Predictor feature table and statistics
|   |-- bootstrap_results.json             # Bootstrap 95% CI analysis
|   |-- supplementary_results.json         # Permutation and per-seed diagnostics
|   |-- composability_ood_results.json     # Composability experiment
|   |-- sym_aug_composability_ood.json     # Symmetry-augmented composability
|   `-- conditioned/                       # Per-condition generation results
`-- requirements.txt
```

## Quick Start

### Option A: Reproduce analysis from saved results

All saved experimental results are included in `outputs/`.

```bash
pip install -r requirements.txt
python scripts/regime_predictor.py
python scripts/bootstrap_regime_map.py
python scripts/supplementary_analysis.py
python generate_scatter.py
```

### Option B: Train from scratch

```bash
pip install -r requirements.txt

# 1. Prepare dataset (requires raw data in data/raw/)
python scripts/prepare_dataset.py

# 2. Train VQ-VAE (100K steps)
python scripts/train_vqvae.py

# 3. Train conditioned AR transformer (80K steps)
python scripts/train_ar_conditioned.py

# 4. Run regime map evaluation
python scripts/test_regime_map.py

# 5. Run CFG sensitivity analysis
python scripts/test_cfg_sensitivity.py

# 6. Fit predictor and supplementary analyses
python scripts/regime_predictor.py
python scripts/bootstrap_regime_map.py
python scripts/supplementary_analysis.py
```

## Data Availability

This repository does not redistribute the raw or processed Minecraft structure
training data. The experiments combine third-party datasets, and some upstream
sources do not provide clear redistribution licenses for repackaged data. To
train from scratch, obtain the upstream data separately and place it under
`data/raw/` before running `scripts/prepare_dataset.py`.

Trained checkpoints are not committed to Git because the final VQ-VAE and AR
Transformer checkpoints are large (~455 MB and ~464 MB). The saved JSON outputs
needed to reproduce the paper's tables and figures are included in `outputs/`;
checkpoints can be distributed separately as release assets if needed.

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- GPU recommended for training and fresh generation runs
- See [requirements.txt](requirements.txt) for dependencies

## Paper

*Which Structural Constraints Are Learnable? A Regime Map for a Minecraft Voxel Generator*

Alex Chengyu Li, 2026

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20821894.svg)](https://doi.org/10.5281/zenodo.20821894)

## Citation

```bibtex
@misc{li2026regimemap,
  author = {Li, Alex Chengyu},
  title = {Which Structural Constraints Are Learnable? A Regime Map for a Minecraft Voxel Generator},
  year = {2026},
  note = {Preprint}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- 3D-Craft dataset by Chen et al. (CVPR 2019)
- rom1504 Minecraft schematics dataset
- VQ-VAE architecture based on van den Oord et al. (NeurIPS 2017)
