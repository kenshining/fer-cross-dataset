# fer-cross-dataset

Implementation code for **"Beyond Single-Dataset Performance: A Controlled Benchmark and Diagnostic Study of Cross-Dataset Generalization in Facial Expression Recognition"** (The Visual Computer).

## Environment

- Python 3.10+, PyTorch 2.x (CUDA recommended)
- Install: `pip install torch torchvision numpy scipy matplotlib pillow`

## Datasets

Five publicly available FER datasets. Download from official sources and place under `data/`:

| Dataset | URL | Access |
|---------|-----|--------|
| AffectNet | https://mohammadmahoor.com/pages/databases/affectnet/ | Registration + license |
| RAF-DB | http://whdeng.cn/RAF/model1.html | Email application |
| FER2013 | https://www.kaggle.com/datasets/msambare/fer2013 | Kaggle account |
| CK+ | https://zenodo.org/records/11221351 | Open access |
| JAFFE | https://zenodo.org/records/14974867 | Open access |

## Random Seeds

- RAF-DB source: 42, 123
- FER2013 source: 42, 123, 456

## Key Scripts

### Training
| Script | Purpose |
|--------|---------|
| `scripts/train_cross_source.py` | Cross-source: FER2013 -> all targets (ResNet, SCN, MHAN) |
| `scripts/train_face_ablation.py` | Ablation: ImageNet vs. MS1MV3 ArcFace pretrained ResNet-18 |
| `scripts/train_vit_baseline.py` | ViT-B/16 Transformer baseline (editor-requested comparison) |
| `scripts/train_clip_baseline.py` | CLIP ViT-B/32 full fine-tuning baseline |
| `scripts/eval_clip_zeroshot.py` | CLIP ViT-B/32 zero-shot evaluation (no training required) |
| `scripts/train_scn_baseline.py` | SCN baseline on RAF-DB |
| `scripts/train_rul_baseline.py` | RUL baseline on RAF-DB |
| `scripts/train_mhan_baseline.py` | MHAN baseline on RAF-DB |
| `scripts/train_aug_baselines.py` | RandAugment / MixUp baselines |

### Analysis & Figures
| Script | Purpose |
|--------|---------|
| `scripts/frequency_signature_analysis.py` | DWT frequency signature analysis (ANOVA + KS tests) |
| `scripts/h2_correlation_analysis.py` | Frequency-generalization correlation (15-point) |
| `scripts/regenerate_fig4_15points.py` | Regenerate Figure 4 with 15-point extended analysis |
| `scripts/scn_alpha_analysis.py` | SCN self-attention weight analysis |
| `scripts/bootstrap_ci.py` | Bootstrap 95% CIs (10,000 resamples) |
| `scripts/statistical_tests.py` | Pairwise statistical comparisons |
| `scripts/analysis_figures.py` | Figures 1-3 (heatmap, bars, generalization gap) |
| `scripts/tsne_analysis.py` | t-SNE feature visualization |

### Core Library (`src/`)
| Module | Purpose |
|--------|---------|
| `models.py` | Backbone builders (ImageNet ResNet-18 + MS1MV3 ArcFace iresnet18) |
| `iresnet.py` | InsightFace iresnet18 architecture |
| `scn_model.py` | SCN (Self-Cure Network) with rank regularization + relabeling |
| `rul_model.py` | RUL (Relative Uncertainty Learning) with feature mixup |
| `train.py` | Training loop + cross-domain evaluation |
| `wavelet.py` | db4 DWT decomposition |
| `preprocess.py` | Face detection / preprocessing |
| `dataset_registry.py` + `dataset_*.py` | Unified dataset interface for 5 datasets |

## Quick Start

**1. Main cross-source experiment:**
```bash
python scripts/train_cross_source.py
```

**2. Face pretraining ablation:**
```bash
# Requires pretrained/ms1mv3_arcface_r18.pth (download from InsightFace Model Zoo)
python scripts/train_face_ablation.py --source rafdb
```

**3. Transformer & VLM baselines:**
```bash
python scripts/train_vit_baseline.py --source rafdb
pip install transformers  # required for CLIP
python scripts/train_clip_baseline.py --source rafdb
python scripts/eval_clip_zeroshot.py  # zero-shot (no training)
```

**4. Frequency analysis:**
```bash
python scripts/frequency_signature_analysis.py
python scripts/h2_correlation_analysis.py
```

**5. Regenerate figures:**
```bash
python scripts/regenerate_fig4_15points.py
python scripts/analysis_figures.py
```

**6. Bootstrap CIs:**
```bash
python scripts/bootstrap_ci.py
```

## Pretrained Weights

MS1MV3 ArcFace iresnet18 `backbone.pth` from [InsightFace Model Zoo](https://github.com/deepinsight/insightface). Place under `pretrained/ms1mv3_arcface_r18.pth`.

## License

MIT
