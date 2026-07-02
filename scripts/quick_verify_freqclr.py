"""
FreqCLR 快速验证: RAF-DB (子集) → FER2013 (跨域)

三组对照:
  1. Baseline: ResNet-18, 标准训练
  2. FreqAug: + 频率振幅扰动 (无对比损失)
  3. FreqCLR: + 频率振幅扰动 + SupCon 对比损失 (我们的方法)

每组 2 seeds, 共 6 次训练. 预计 2-3 小时.

用法:
  python fer_wavelet/scripts/quick_verify_freqclr.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision.models import resnet18, ResNet18_Weights

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.augment import FreqPerturb
from src.loss import SupConLoss
from src.wavelet import batch_dwt_torch
from src.preprocess import center_crop_resize, pil_to_tensor01

# ---- 配置 ----
DATA_ROOT = Path("e:/scientific/小波/data")
RUNS_ROOT = _REPO / "runs" / "freqclr_quick_verify"
os.makedirs(RUNS_ROOT, exist_ok=True)

BATCH_SIZE = 16
FACE_SIZE = 224
NUM_CLASSES = 7
EPOCHS = 30
LR = 1e-3
TRAIN_SAMPLES = 2800  # 400/类 × 7 = 2800
N_SEEDS = 2
SIGMA_FREQ = 0.15  # 频率扰动标准差
SUPCON_TEMP = 0.07
SUPCON_WEIGHT = 0.3  # 对比损失权重


# ====================================================================
# 模型
# ====================================================================

class FreqCLRModel(nn.Module):
    """ResNet-18 + 投影头 (用于对比学习) + 分类头."""

    def __init__(self, num_classes: int = 7, proj_dim: int = 128):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        backbone = resnet18(weights=weights)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])  # → (B,512,1,1)
        self.feat_dim = 512

        # 投影头 (用于对比学习)
        self.projection = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(),
            nn.Linear(self.feat_dim, proj_dim),
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor, return_proj: bool = False):
        f = self.encoder(x)  # (B, 512, 1, 1)
        f_flat = f.view(f.size(0), -1)  # (B, 512)
        logits = self.classifier(f)

        if return_proj:
            proj = self.projection(f_flat)
            proj = F.normalize(proj, dim=1)
            return logits, proj
        return logits


# ====================================================================
# 数据
# ====================================================================

def build_rafdb_subset(n_per_class: int, seed: int):
    """从 RAF-DB 训练集每类抽样构建子集。"""
    from src.dataset_registry import REGISTRY
    import random as _random

    info = REGISTRY["rafdb"]
    full_ds = info["dataset_cls"](DATA_ROOT / "RAF-DB", split="train")

    rng = _random.Random(seed)
    class_indices = {c: [] for c in range(NUM_CLASSES)}
    for idx in range(len(full_ds)):
        _, label = full_ds[idx]
        lbl = int(label) if not isinstance(label, int) else label
        if lbl in class_indices:
            class_indices[lbl].append(idx)

    selected = []
    for c in range(NUM_CLASSES):
        indices = class_indices[c]
        n = min(n_per_class, len(indices))
        selected.extend(rng.sample(indices, n))
    rng.shuffle(selected)
    return Subset(full_ds, selected)


def build_fer2013_val():
    """构建 FER2013 验证集 DataLoader (跨域测试)。"""
    from src.dataset_registry import REGISTRY
    info = REGISTRY["fer2013"]
    _, val_loader = _create_loaders_fer(info, BATCH_SIZE)
    return val_loader


def _create_loaders_fer(info, batch_size):
    """手动构建 FER2013 DataLoader。"""
    import csv
    import random
    from PIL import Image

    csv_path = DATA_ROOT / "Fer2013" / "fer2013" / "fer2013.csv"
    if not csv_path.exists():
        csv_path = DATA_ROOT / "Fer2013" / "icml_face_data.csv"

    class PixelDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.samples = []
            usage_map = {"Training": 0, "PublicTest": 1, "PrivateTest": 2}
            with open(csv_path) as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if len(row) < 3:
                        continue
                    try:
                        emotion = int(row[0])
                        pixels = np.array(row[1].split(), dtype=np.uint8)
                        usage = row[2].strip()
                    except (ValueError, IndexError):
                        continue
                    if usage_map.get(usage, -1) == 1:  # PublicTest
                        self.samples.append((emotion, pixels))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            emotion, pixels = self.samples[idx]
            img = pixels.reshape(48, 48)
            img_pil = Image.fromarray(img).convert("RGB").resize(
                (FACE_SIZE, FACE_SIZE), Image.BILINEAR
            )
            tensor = pil_to_tensor01(img_pil)
            return tensor, emotion

    ds = PixelDataset()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return None, loader


# ====================================================================
# 评估
# ====================================================================

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    tp = torch.zeros(NUM_CLASSES, device=device)
    fp = torch.zeros(NUM_CLASSES, device=device)
    fn = torch.zeros(NUM_CLASSES, device=device)

    for batch_data, labels in loader:
        labels = labels.to(device)
        if isinstance(batch_data, torch.Tensor):
            rgb = batch_data.to(device)
        else:
            rgb_list = []
            for pil in batch_data:
                crop = center_crop_resize(pil.convert("RGB"), FACE_SIZE)
                rgb_list.append(pil_to_tensor01(crop))
            rgb = torch.stack(rgb_list, dim=0).to(device)

        logits = model(rgb)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        for c in range(NUM_CLASSES):
            tp[c] += ((pred == c) & (labels == c)).sum()
            fp[c] += ((pred == c) & (labels != c)).sum()
            fn[c] += ((pred != c) & (labels == c)).sum()

    acc = correct / max(total, 1)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    macro_f1 = f1.mean().item()
    return {"acc": acc, "macro_f1": macro_f1}


# ====================================================================
# 训练
# ====================================================================

def train_baseline(model, train_loader, optimizer, criterion, device, epoch, freq_perturb=None):
    """标准训练 (可选 FreqAug 数据增强)。"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for rgb, labels in train_loader:
        rgb = rgb.to(device)
        labels = labels.to(device)

        # 频率扰动 (FreqAug)
        if freq_perturb is not None:
            rgb = freq_perturb(rgb)

        optimizer.zero_grad()
        logits = model(rgb)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def train_freqclr(model, train_loader, optimizer, ce_criterion, supcon_loss, device, epoch):
    """FreqCLR 训练: 双前向传播 + CE + SupCon。"""
    model.train()
    total_ce = 0.0
    total_supcon = 0.0
    correct = 0
    total = 0

    freq_perturb = FreqPerturb(sigma=SIGMA_FREQ, p=1.0)

    for rgb, labels in train_loader:
        rgb = rgb.to(device)
        labels = labels.to(device)
        bs = rgb.size(0)

        # 生成两个频率扰动视角
        rgb_v1 = freq_perturb(rgb)  # (B, 3, H, W)
        rgb_v2 = freq_perturb(rgb)  # 独立扰动

        # 拼接: [v1_batch, v2_batch]
        rgb_cat = torch.cat([rgb_v1, rgb_v2], dim=0)  # (2B, 3, H, W)

        optimizer.zero_grad()

        # 前向传播
        logits, proj = model(rgb_cat, return_proj=True)  # logits: (2B, 7), proj: (2B, 128)

        # 分类损失 (对两个视角分别计算)
        labels_cat = labels.repeat(2)
        ce_loss = ce_criterion(logits, labels_cat)

        # 监督对比损失
        sc_loss = supcon_loss(proj, labels)  # 内部处理视图

        # 联合损失
        loss = ce_loss + SUPCON_WEIGHT * sc_loss
        loss.backward()
        optimizer.step()

        total_ce += ce_loss.item() * bs
        total_supcon += sc_loss.item() * bs
        pred = logits[:bs].argmax(dim=1)  # 仅用 view1 统计准确率
        correct += (pred == labels).sum().item()
        total += bs

    return total_ce / total, total_supcon / total, correct / total


def build_rafdb_loader(subset, batch_size, shuffle=True):
    """构建 RAF-DB PIL 图像的 DataLoader (转为 tensor)。"""
    from torch.utils.data import DataLoader

    class TensorWrapper(torch.utils.data.Dataset):
        def __init__(self, subset_ds):
            self.ds = subset_ds

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            pil_img, label = self.ds[idx]
            crop = center_crop_resize(pil_img.convert("RGB"), FACE_SIZE)
            tensor = pil_to_tensor01(crop)
            return tensor, label

    wrapped = TensorWrapper(subset)
    return DataLoader(wrapped, batch_size=batch_size, shuffle=shuffle, num_workers=0)


# ====================================================================
# 主函数
# ====================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print("=" * 70)
    print("FreqCLR 快速验证: RAF-DB → FER2013")
    print(f"训练样本: {TRAIN_SAMPLES} ({TRAIN_SAMPLES//NUM_CLASSES}/类)")
    print(f"Seeds: {N_SEEDS} × 3 方法 = {N_SEEDS*3} 次训练")
    print("=" * 70)

    # 构建跨域测试集
    print("\n[准备] 构建 FER2013 跨域测试集...")
    fer_loader = build_fer2013_val()
    print(f"  FER2013 测试样本: {len(fer_loader.dataset)}")

    # 构建 RAF-DB 域内验证集 (使用 test split)
    from src.dataset_registry import REGISTRY
    raf_info = REGISTRY["rafdb"]
    raf_val_ds = raf_info["dataset_cls"](DATA_ROOT / "RAF-DB", split="test")

    class ValTensorWrapper(torch.utils.data.Dataset):
        def __init__(self, ds):
            self.ds = ds
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            pil_img, label = self.ds[idx]
            pil_img = pil_img.convert("RGB") if hasattr(pil_img, 'convert') else pil_img
            crop = center_crop_resize(pil_img, FACE_SIZE)
            tensor = pil_to_tensor01(crop)
            return tensor, label

    raf_val_wrapped = ValTensorWrapper(raf_val_ds)
    raf_val_loader = DataLoader(raf_val_wrapped, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"  RAF-DB 域内测试样本: {len(raf_val_loader.dataset)}")

    all_results = []

    for seed in [42, 123][:N_SEEDS]:
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 构建训练子集
        subset = build_rafdb_subset(TRAIN_SAMPLES // NUM_CLASSES, seed)
        train_loader = build_rafdb_loader(subset, BATCH_SIZE, shuffle=True)
        print(f"\n{'='*50}")
        print(f"Seed {seed}: 训练样本 {len(subset)}")
        print(f"{'='*50}")

        for method in ["Baseline", "FreqAug", "FreqCLR"]:
            run_tag = f"{method}_seed{seed}"
            print(f"\n  [{run_tag}]", flush=True)

            model = FreqCLRModel(num_classes=NUM_CLASSES).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6,
            )
            ce_criterion = nn.CrossEntropyLoss()
            supcon_loss = SupConLoss(temperature=SUPCON_TEMP)

            freq_perturb = FreqPerturb(sigma=SIGMA_FREQ, p=1.0) if method in ("FreqAug", "FreqCLR") else None

            best_in_f1 = 0.0
            best_state = None
            hist = []

            t0 = time.time()
            for epoch in range(1, EPOCHS + 1):
                if method == "FreqCLR":
                    ce_l, sc_l, train_acc = train_freqclr(
                        model, train_loader, optimizer, ce_criterion, supcon_loss, device, epoch,
                    )
                else:
                    ce_l, train_acc = train_baseline(
                        model, train_loader, optimizer, ce_criterion, device, epoch,
                        freq_perturb=freq_perturb,
                    )
                    sc_l = 0.0

                # 域内评估
                in_metrics = evaluate(model, raf_val_loader, device)
                scheduler.step(in_metrics["macro_f1"])

                if in_metrics["macro_f1"] > best_in_f1:
                    best_in_f1 = in_metrics["macro_f1"]
                    best_state = deepcopy(model.state_dict())

                hist.append({
                    "epoch": epoch,
                    "ce_loss": ce_l,
                    "sc_loss": sc_l,
                    "train_acc": train_acc,
                    "in_acc": in_metrics["acc"],
                    "in_f1": in_metrics["macro_f1"],
                })

                if epoch % 10 == 0 or epoch == 1:
                    print(f"    Epoch {epoch:2d}: ce={ce_l:.4f}" +
                          (f", sc={sc_l:.4f}" if method == "FreqCLR" else "") +
                          f", in_f1={in_metrics['macro_f1']:.4f}",
                          flush=True)

            elapsed = time.time() - t0

            # 加载最佳模型
            model.load_state_dict(best_state)
            model.eval()

            # 域内最终
            in_final = evaluate(model, raf_val_loader, device)
            # 跨域评估
            cross_final = evaluate(model, fer_loader, device)

            result = {
                "method": method,
                "seed": seed,
                "best_in_f1": best_in_f1,
                "in_f1": in_final["macro_f1"],
                "in_acc": in_final["acc"],
                "cross_f1": cross_final["macro_f1"],
                "cross_acc": cross_final["acc"],
                "gen_drop": in_final["macro_f1"] - cross_final["macro_f1"],
                "elapsed_min": elapsed / 60,
            }
            all_results.append(result)

            print(f"    域内 F1={in_final['macro_f1']:.4f}, "
                  f"跨域 F1={cross_final['macro_f1']:.4f}, "
                  f"泛化下降={result['gen_drop']:.4f}, "
                  f"耗时={elapsed/60:.1f}min",
                  flush=True)

            # 释放显存
            del model, optimizer, scheduler
            torch.cuda.empty_cache()

    # ---- 汇总 ----
    print(f"\n{'=' * 70}")
    print("快速验证结果汇总")
    print("=" * 70)

    for method in ["Baseline", "FreqAug", "FreqCLR"]:
        method_results = [r for r in all_results if r["method"] == method]
        if not method_results:
            continue
        cross_f1s = [r["cross_f1"] for r in method_results]
        in_f1s = [r["in_f1"] for r in method_results]
        drops = [r["gen_drop"] for r in method_results]
        print(f"\n{method}:")
        print(f"  跨域 F1: {np.mean(cross_f1s):.4f} ± {np.std(cross_f1s):.4f}  [{', '.join(f'{x:.4f}' for x in cross_f1s)}]")
        print(f"  域内 F1: {np.mean(in_f1s):.4f} ± {np.std(in_f1s):.4f}")
        print(f"  泛化下降: {np.mean(drops):.4f} ± {np.std(drops):.4f}")

    # 判定
    baseline_cross = np.mean([r["cross_f1"] for r in all_results if r["method"] == "Baseline"])
    freqaug_cross = np.mean([r["cross_f1"] for r in all_results if r["method"] == "FreqAug"])
    freqclr_cross = np.mean([r["cross_f1"] for r in all_results if r["method"] == "FreqCLR"])

    from scipy import stats
    b_vals = [r["cross_f1"] for r in all_results if r["method"] == "Baseline"]
    fc_vals = [r["cross_f1"] for r in all_results if r["method"] == "FreqCLR"]

    if len(b_vals) >= 2 and len(fc_vals) >= 2:
        t_stat, p_val = stats.ttest_ind(fc_vals, b_vals)
        pooled = np.sqrt((np.var(b_vals) + np.var(fc_vals)) / 2)
        d = (freqclr_cross - baseline_cross) / max(pooled, 1e-8)
        print(f"\nFreqCLR vs Baseline: Δ={freqclr_cross-baseline_cross:+.4f}, t={t_stat:.3f}, p={p_val:.4f}, d={d:.3f}")

    print(f"\n结论: ", end="")
    if freqclr_cross > freqaug_cross and freqclr_cross > baseline_cross:
        print("✅ FreqCLR 优于 Baseline 和 FreqAug — 方向可行！")
    elif freqclr_cross > baseline_cross:
        print("⚠️ FreqCLR 优于 Baseline 但未超过 FreqAug — 对比损失贡献不足")
    else:
        print("❌ FreqCLR 未优于 Baseline — 需要调整方案")

    # 保存
    report_path = RUNS_ROOT / "verify_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {report_path}")


if __name__ == "__main__":
    main()
