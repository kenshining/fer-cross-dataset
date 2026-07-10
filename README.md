# fer-cross-dataset

Implementation code for **"Beyond Single-Dataset Performance: A Controlled Benchmark and Diagnostic Study of Cross-Dataset Generalization in Facial Expression Recognition"** (SPIE Proceedings, 2026).

## Overview

This repository provides a reproducible benchmark for cross-dataset generalization in Facial Expression Recognition (FER). Eight methods spanning four methodological categories are evaluated under a leave-one-dataset-out protocol across five FER datasets, with 10 independent training seeds per method for statistical reliability.

## Environment

- Python 3.10+, PyTorch 2.x (CUDA recommended)
- Install: `pip install torch torchvision numpy scipy matplotlib pillow pywt`

## Datasets

Five publicly available FER datasets. Download from official sources and place under `data/`:

| Dataset | URL | Access |
|---------|-----|--------|
| AffectNet | https://mohammadmahoor.com/pages/databases/affectnet/ | Registration + license |
| RAF-DB | http://whdeng.cn/RAF/model1.html | Email application |
| FER2013 | https://www.kaggle.com/datasets/msambare/fer2013 | Kaggle account |
| CK+ | https://zenodo.org/records/11221351 | Open access |
| JAFFE | https://zenodo.org/records/14974867 | Open access |

## Methods Evaluated

| Category | Method | Backbone | Seeds |
|----------|--------|----------|:---:|
| Baseline | ResNet-18 | ResNet-18 (ImageNet) | 10 |
| FER-specific | SCN (Self-Cure Network) | ResNet-18 | 10 |
| FER-specific | RUL (Relative Uncertainty Learning) | ResNet-18 | 10 |
| FER-specific | MHAN (Multi-Head Hybrid Attention) | MixedFeatureNet (MS-Celeb-1M) | 10 |
| Data augmentation | RandAugment | ResNet-18 | 10 |
| Data augmentation | MixUp | ResNet-18 | 10 |
| Transformer | ViT-B/16 | ViT-B/16 (ImageNet-21k) | 10 |
| Domain generalization | SWAD | ResNet-18 | 10 |

## Random Seeds

All methods trained with 10 random seeds: **10, 42, 89, 123, 456, 781, 789, 999, 1337, 2026**.

## Repository Structure

```
fer-cross-dataset/
├── src/                    # Core library
│   ├── dataset_registry.py # Unified dataset interface
│   ├── dataset_*.py        # Per-dataset loaders
│   ├── models.py           # Backbone builders
│   ├── iresnet.py          # ArcFace iresnet18
│   ├── scn_model.py        # SCN with rank regularization
│   ├── rul_model.py        # RUL with feature mixup
│   ├── wavelet.py          # Configurable DWT decomposition
│   ├── preprocess.py       # Face detection / preprocessing
│   ├── train.py            # Training loop
│   └── color_palette.py    # Unified color palette
├── scripts/                # Training and analysis scripts
│   ├── run_final_10seeds.py     # Main 10-seed training (all methods)
│   ├── run_all_seeds.py         # Batch training for core methods
│   ├── run_scn_10seeds.py       # SCN-specific training
│   ├── run_rul_10seeds.py       # RUL-specific training
│   ├── run_swad_10seeds.py      # SWAD training
│   ├── run_mhan_vit_fix.py      # MHAN + ViT-B training
│   ├── run_ablation.py          # Face pretraining ablation
│   ├── wavelet_sensitivity.py   # 7-wavelet sensitivity analysis
│   ├── multi_factor_analysis.py # 6-feature Spearman correlation
│   ├── scn_alpha_posthoc.py     # SCN alpha post-hoc analysis
│   ├── generate_figures.py      # Revision figures (5 new)
│   ├── redraw_fig1_fig2.py      # Updated Fig 1 + Fig 2
│   ├── frequency_signature_analysis.py # DWT frequency signature
│   ├── freq_generalization_cross.py    # Frequency-generalization correlation
│   ├── scn_alpha_analysis.py     # SCN alpha weight analysis
│   ├── bootstrap_ci.py           # Bootstrap confidence intervals
│   ├── statistical_tests.py      # Pairwise statistical comparisons
│   ├── analysis_figures.py       # Original figures (Fig 1-3)
│   └── tsne_analysis.py          # t-SNE feature visualization
├── runs/                  # Experiment outputs
│   ├── all_10seeds/       # Main 10-seed results
│   ├── wavelet_sensitivity/    # Wavelet analysis results
│   ├── multi_factor/           # Multi-factor Spearman results
│   ├── scn_alpha_posthoc/      # SCN alpha post-hoc results
│   └── figures/           # Generated figures
└── pretrained/            # Pretrained weights
    └── ms1mv3_arcface_r18.pth  # MS1MV3 ArcFace
```

## Quick Start

**1. Main 10-seed training (all 8 methods):**
```bash
python scripts/run_final_10seeds.py
```

**2. Analysis scripts:**
```bash
# Wavelet sensitivity (7 bases)
python scripts/wavelet_sensitivity.py

# Multi-factor dataset analysis
python scripts/multi_factor_analysis.py

# SCN alpha post-hoc
python scripts/scn_alpha_posthoc.py

# Frequency signature analysis
python scripts/frequency_signature_analysis.py
```

**3. Generate revision figures:**
```bash
# New figures (full ranking, wavelet, SCN alpha, multi-factor, MHAN vs ResNet)
python scripts/generate_figures.py

# Updated Fig 1 (heatmap) + Fig 2 (bar chart) with 8 methods
python scripts/redraw_fig1_fig2.py
```

## Key Results (10-Seed, RAF-DB Source)

| Method | Mean Macro-F1 | Std | CV | p vs ResNet | Cohen's d |
|--------|:---:|:---:|:---:|:---:|:---:|
| SWAD | 0.289 | 0.009 | 3.2% | 0.001 | 2.24 |
| MHAN | 0.283 | 0.024 | 8.6% | 0.011 | 1.35 |
| MixUp | 0.272 | 0.020 | 7.5% | 0.056 | 1.01 |
| RUL | 0.272 | 0.016 | 5.9% | 0.040 | 1.10 |
| SCN | 0.270 | 0.010 | 3.6% | 0.049 | 1.11 |
| RandAug | 0.264 | 0.022 | 8.5% | 0.212 | 0.63 |
| ViT-B/16 | 0.264 | 0.066 | 25.1% | 0.594 | 0.25 |
| ResNet-18 | 0.252 | 0.022 | 8.6% | — | — |

Statistical significance assessed via Welch's t-test. Complete pairwise comparisons in the manuscript (Table 3).

## Diagnostic Findings

- **Frequency-generalization correlation**: r = 0.70 (p = 0.004, n = 15), robust across 7 wavelet bases (all p < 0.05)
- **SCN alpha domain-shift detection**: 8.0% mean alpha reduction on cross-domain samples; above-median alpha samples achieve +0.080 accuracy delta
- **Training stability**: CNN methods CV 3.2-8.6%; ViT-B/16 CV 25.1%

## License

MIT
