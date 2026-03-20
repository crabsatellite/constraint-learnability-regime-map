# Which Structural Constraints Are Learnable?

Code, experimental results, and [interactive explainer](https://crabsatellite.github.io/constraint-learnability-regime-map/) for *Which Structural Constraints Are Learnable? A Regime Map for a Minecraft Voxel Generator*.

## Key Results

| Regime | Properties | Ctrl% Range | Bottleneck |
|--------|-----------|-------------|------------|
| Controllable | 9 / 14 | >100% | None (standard CFG works) |
| Approachable | 4 / 14 | 20--100% | Frequency floor (more data or stronger guidance) |
| Unresponsive | 1 / 14 | <20% | Representation ceiling (decoder/latent change needed) |

Composite predictor: effective signal x min(CV, 1) correlates with controllability at **Spearman rho = 0.879** (permutation p = 0.002, n = 10 emergent properties). Neither signal alone (0.624) nor CV alone (0.636) reaches significance; their product does.

## What This Is

- **Three-regime learnability map** across 14 structural properties of Minecraft buildings (10,310 builds, 32^3 voxel grids, ~260 block types)
- **VQ-VAE** (32^3 -> 8^3 latent, 2048 codebook, ~22M params) + **AR Transformer** (512d, 12 layers, 8 heads, ~55M params) with classifier-free guidance
- **Dual bottleneck analysis**: CFG sensitivity (s=0,2,4) separates frequency-limited from representation-limited constraints
- **Predictive framework**: composite score (signal x min(CV, 1)) predicts regime before running generation experiments
- **[Interactive explainer](https://crabsatellite.github.io/constraint-learnability-regime-map/)**: bilingual (EN/ZH) step-by-step walkthrough of the paper for non-specialist audiences

## Repository Structure

```
constraint-learnability/
├── index.html                             # Interactive paper explainer (bilingual EN/ZH)
├── figures/
│   ├── scatter_predictor.png              # Composite score vs controllability (Figure 1)
│   ├── cfg_sensitivity.png                # CFG sensitivity visualization (Figure 2)
│   └── condition_comparison.png           # Condition comparison visualization
├── models/
│   ├── vqvae.py                           # VQ-VAE (3D conv encoder-decoder, 2048 codebook)
│   └── ar_transformer.py                  # AR Transformer with structural conditioning + CFG
├── scripts/
│   ├── prepare_dataset.py                 # Unified dataset preparation (text2mc, 3D-Craft, rom1504)
│   ├── dataset.py                         # PyTorch dataset for VQ-VAE training
│   ├── dataset_dense.py                   # Dense latent code dataset for AR training
│   ├── train_vqvae.py                     # VQ-VAE training (100K steps)
│   ├── train_ar.py                        # Unconditional AR training
│   ├── train_ar_conditioned.py            # Conditioned AR training (80K steps, CFG)
│   ├── extract_structural_features.py     # Extract 14 structural properties from builds
│   ├── eval_structural.py                 # Evaluate controllability across conditions
│   ├── test_regime_map.py                 # Full regime map evaluation (5 seeds x 8 samples)
│   ├── test_cfg_sensitivity.py            # CFG sensitivity sweep (s=0,2,4)
│   ├── test_conditioning.py               # Per-condition generation tests
│   ├── test_composability_ood.py          # Composability and OOD experiments
│   ├── test_robustness.py                 # Threshold robustness analysis
│   ├── regime_predictor.py                # Decision tree predictor + statistical analysis
│   ├── bootstrap_regime_map.py            # Bootstrap 95% CI computation
│   ├── generate.py                        # Generate builds from trained models
│   ├── generate_conditioned.py            # Conditioned generation
│   ├── augment_symmetry.py                # Symmetry data augmentation
│   ├── run_symmetry_experiment.py         # Symmetry augmentation experiments
│   ├── visualize_structures.py            # Visualization utilities
│   └── legacy_block_map.py               # Minecraft legacy block ID mapping
├── outputs/
│   ├── regime_map_results.json            # Table 1 data (14 properties, 5 seeds)
│   ├── cfg_sensitivity_results.json       # Table 2 data (CFG s=0,2,4)
│   ├── regime_predictor_results.json      # Predictor statistics and LOO results
│   ├── bootstrap_results.json             # Bootstrap 95% CI analysis
│   ├── robustness_results.json            # Threshold robustness sweep
│   ├── composability_ood_results.json     # Composability experiment
│   ├── sym_aug_composability_ood.json     # Symmetry-augmented composability
│   └── conditioned/                       # Per-condition generation results
└── requirements.txt
```

## Quick Start

### Option A: Reproduce analysis from saved results (~1 minute)

All experimental results are included in `outputs/`. Run the predictor analysis directly:

```bash
pip install -r requirements.txt
python scripts/regime_predictor.py
```

### Option B: Train from scratch (~24 hours on 1 GPU)

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

# 6. Fit predictor
python scripts/regime_predictor.py
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- ~8 GB VRAM for training
- See [requirements.txt](requirements.txt) for full dependencies

## Paper

*Which Structural Constraints Are Learnable? A Regime Map for a Minecraft Voxel Generator*
Alex Chengyu Li, 2026

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19135431.svg)](https://doi.org/10.5281/zenodo.19135431)

## Citation

```bibtex
@misc{li2026regimemap,
  author    = {Li, Alex Chengyu},
  title     = {Which Structural Constraints Are Learnable?
               A Regime Map for a Minecraft Voxel Generator},
  year      = {2026},
  note      = {Preprint}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- 3D-Craft dataset by Chen et al. (CVPR 2019)
- rom1504 Minecraft schematics dataset
- VQ-VAE architecture based on van den Oord et al. (NeurIPS 2017)
