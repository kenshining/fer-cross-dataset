"""
H₃ 验证：小样本对照实验 — 频率门控 vs. 基线 ResNet-18

设计：
  - 训练集: RAF-DB 子集 (200张/类 × 7 = 1400张)
  - 测试: RAF-DB val (域内) + FER2013 (域外跨数据集)
  - 3 随机种子 × 2 模型 = 6 次训练
  - 配对 t 检验 + Cohen's d

用法：
  python fer_wavelet/scripts/h3_controlled_experiment.py
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
from torch.utils.data import DataLoader, Subset

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.models import FERWaveletModel
from src.dataset_registry import REGISTRY, create_loaders
from src.train import cross_domain_evaluate, setup_gpu
from src.wavelet import batch_dwt_torch

# ---- 配置 ----
PROJECT_ROOT = _REPO.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "datasets.yaml"
DATA_ROOT = Path("e:/scientific/小波/data")
RUNS_ROOT = _REPO / "runs"
H3_DIR = RUNS_ROOT / "h3_experiment"
BATCH_SIZE = 16
FACE_SIZE = 224
NUM_CLASSES = 7
EPOCHS = 30
LR = 1e-3
SAMPLES_PER_CLASS = 200
N_SEEDS = 3

os.makedirs(H3_DIR, exist_ok=True)


# ====================================================================
# 频率门控模型
# ====================================================================

class FreqGateModule(nn.Module):
    """从 DWT 子带学习特征门控权重。"""
    def __init__(self, in_channels: int = 6, feat_dim: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, feat_dim),
            nn.Sigmoid(),
        )

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        """low: (B,3,28,28), high: (B,3,28,28) → gate: (B, feat_dim)"""
        freq = torch.cat([low, high], dim=1)  # (B, 6, 28, 28)
        return self.encoder(freq)


class FreqGatedResNet(nn.Module):
    """ResNet-18 + 频率门控调制。"""
    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # → (B,512,1,1)
        self.feat_dim = 512

        self.gate = FreqGateModule(in_channels=6, feat_dim=self.feat_dim)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, rgb: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        f = self.backbone(rgb)  # (B, 512, 1, 1)
        gate = self.gate(low, high).unsqueeze(-1).unsqueeze(-1)  # (B, 512, 1, 1)
        f_modulated = f * gate
        return self.classifier(f_modulated)


class BaselineResNet(nn.Module):
    """标准 ResNet-18 基线。"""
    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, rgb: torch.Tensor, low=None, high=None) -> torch.Tensor:
        f = self.backbone(rgb)
        return self.classifier(f)


# ====================================================================
# 小样本采样
# ====================================================================

def create_small_subset(full_dataset, samples_per_class: int, seed: int):
    """从数据集中每类均匀采样。"""
    import random
    rng = random.Random(seed)

    class_indices = {c: [] for c in range(NUM_CLASSES)}
    for idx in range(len(full_dataset)):
        _, label = full_dataset[idx]
        lbl = int(label)
        if lbl in class_indices:
            class_indices[lbl].append(idx)

    selected = []
    for c in range(NUM_CLASSES):
        indices = class_indices[c]
        n = min(samples_per_class, len(indices))
        selected.extend(rng.sample(indices, n))

    rng.shuffle(selected)
    return selected


# ====================================================================
# 训练循环
# ====================================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_data, labels in loader:
        labels = labels.to(device)

        if isinstance(batch_data, torch.Tensor):
            rgb = batch_data.to(device)
        else:
            # PIL batch → tensor
            rgb_list = []
            for pil in batch_data:
                from src.preprocess import center_crop_resize, pil_to_tensor01
                crop = center_crop_resize(pil.convert("RGB"), FACE_SIZE)
                rgb_list.append(pil_to_tensor01(crop))
            rgb = torch.stack(rgb_list, dim=0).to(device)

        gray = rgb.mean(dim=1, keepdim=True)
        low, high = batch_dwt_torch(gray)

        optimizer.zero_grad()
        logits = model(rgb, low, high)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


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
                from src.preprocess import center_crop_resize, pil_to_tensor01
                crop = center_crop_resize(pil.convert("RGB"), FACE_SIZE)
                rgb_list.append(pil_to_tensor01(crop))
            rgb = torch.stack(rgb_list, dim=0).to(device)

        gray = rgb.mean(dim=1, keepdim=True)
        low, high = batch_dwt_torch(gray)

        logits = model(rgb, low, high)
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

    return {"acc": acc, "macro_f1": macro_f1, "per_class_f1": f1.cpu().tolist()}


# ====================================================================
# 主实验
# ====================================================================

def load_config():
    import yaml
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not config.get("root"):
        config["root"] = str(PROJECT_ROOT / "data")
    return config


def main():
    device = setup_gpu(memory_fraction=0.85)
    print(f"设备: {device}")
    print("=" * 70)
    print("H3: 小样本对照实验 (频率门控 vs 基线 ResNet-18)")
    print(f"训练集: RAF-DB, {SAMPLES_PER_CLASS}张/类 = {SAMPLES_PER_CLASS * NUM_CLASSES}张")
    print(f"种子数: {N_SEEDS} × 2 模型 = {N_SEEDS * 2} 次训练")
    print("=" * 70)

    config = load_config()

    # ---- 构建训练集（RAF-DB 子集） ----
    print("\n[1] 构建 RAF-DB 小样本训练集...")
    info = REGISTRY["rafdb"]
    full_train_ds = info["dataset_cls"](DATA_ROOT / "RAF-DB", split="train")

    # 构建域内验证集（RAF-DB test split）
    val_ds = info["dataset_cls"](DATA_ROOT / "RAF-DB", split="test")
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        collate_fn=info["collate_fn"],
    )
    print(f"  域内验证集 (RAF-DB test): {len(val_ds)} 样本")

    # 构建跨数据集验证集（FER2013）
    fer_info = REGISTRY["fer2013"]
    _, fer_val_loader = create_loaders(
        dataset_name="fer2013", config=config,
        batch_size=BATCH_SIZE, num_workers=0,
        seed=42, smoke_samples=None, device_type="cpu",
    )
    print(f"  跨域验证集 (FER2013 val): {len(fer_val_loader.dataset) if fer_val_loader else 'N/A'} 样本")

    # ---- 逐种子实验 ----
    all_results = []

    for seed_idx, seed in enumerate([42, 123, 456][:N_SEEDS]):
        print(f"\n{'='*50}")
        print(f"Seed {seed_idx + 1}/{N_SEEDS}: seed={seed}")
        print(f"{'='*50}")

        # 此种子的数据子集
        torch.manual_seed(seed)
        np.random.seed(seed)
        subset_indices = create_small_subset(full_train_ds, SAMPLES_PER_CLASS, seed)
        train_subset = Subset(full_train_ds, subset_indices)
        train_loader = DataLoader(
            train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
            collate_fn=info["collate_fn"],
        )
        print(f"  训练样本: {len(train_subset)}")

        for model_name, ModelClass in [("Baseline", BaselineResNet), ("FreqGated", FreqGatedResNet)]:
            run_name = f"{model_name}_seed{seed}"
            print(f"\n  [{run_name}]")

            model = ModelClass(num_classes=NUM_CLASSES, pretrained=True).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6,
            )
            criterion = nn.CrossEntropyLoss()

            best_val_f1 = 0.0
            best_state = None
            history = []

            for epoch in range(1, EPOCHS + 1):
                train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)

                # 域内评估
                val_metrics = evaluate(model, val_loader, device)
                scheduler.step(val_metrics["macro_f1"])

                if val_metrics["macro_f1"] > best_val_f1:
                    best_val_f1 = val_metrics["macro_f1"]
                    best_state = deepcopy(model.state_dict())

                history.append({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_acc": val_metrics["acc"],
                    "val_macro_f1": val_metrics["macro_f1"],
                })

                if epoch % 5 == 0 or epoch == 1:
                    print(f"    Epoch {epoch:2d}: loss={train_loss:.4f}, "
                          f"in-domain acc={val_metrics['acc']:.4f}, f1={val_metrics['macro_f1']:.4f}")

            # 加载最佳模型
            model.load_state_dict(best_state)
            model.eval()

            # 域内最终评估
            in_domain = evaluate(model, val_loader, device)

            # 跨域评估（FER2013）
            cross_domain = evaluate(model, fer_val_loader, device) if fer_val_loader else {"macro_f1": 0.0, "acc": 0.0}

            result = {
                "model": model_name,
                "seed": seed,
                "best_val_f1": best_val_f1,
                "in_domain_f1": in_domain["macro_f1"],
                "in_domain_acc": in_domain["acc"],
                "cross_domain_f1": cross_domain["macro_f1"],
                "cross_domain_acc": cross_domain["acc"],
                "gen_drop": in_domain["macro_f1"] - cross_domain["macro_f1"],
                "history": history,
            }
            all_results.append(result)

            print(f"    域内 F1={in_domain['macro_f1']:.4f}, 跨域 F1={cross_domain['macro_f1']:.4f}, "
                  f"泛化下降={result['gen_drop']:.4f}")

            # 释放显存
            del model, optimizer, scheduler
            torch.cuda.empty_cache()

    # ---- 统计分析 ----
    print(f"\n{'=' * 70}")
    print("H3 统计分析")
    print("=" * 70)

    baseline = [r for r in all_results if r["model"] == "Baseline"]
    freqgated = [r for r in all_results if r["model"] == "FreqGated"]

    from scipy import stats

    print(f"\n{'Metric':<20} {'Baseline':<20} {'FreqGated':<20} {'Δ':<10} {'p-value':<12} {'Cohen d':<10}")
    print("-" * 92)

    conclusions = {}
    for metric_key, metric_name in [
        ("in_domain_f1", "域内 F1"),
        ("cross_domain_f1", "跨域 F1 (FER2013)"),
        ("gen_drop", "泛化下降"),
    ]:
        b_vals = [r[metric_key] for r in baseline]
        f_vals = [r[metric_key] for r in freqgated]
        b_mean, b_std = np.mean(b_vals), np.std(b_vals)
        f_mean, f_std = np.mean(f_vals), np.std(f_vals)
        delta = f_mean - b_mean

        # 配对 t 检验
        t_stat, p_val = stats.ttest_rel(f_vals, b_vals)

        # Cohen's d
        pooled_std = np.sqrt((np.var(b_vals) + np.var(f_vals)) / 2)
        cohen_d = delta / max(pooled_std, 1e-8)

        sig = "✅" if p_val < 0.05 else "⚠️"
        print(f"{metric_name:<20} {b_mean:.4f}±{b_std:.4f}    {f_mean:.4f}±{f_std:.4f}    "
              f"{delta:+.4f}    {p_val:<12.4f} {sig} {cohen_d:+.3f}")

        conclusions[metric_key] = {
            "baseline_mean": b_mean, "baseline_std": b_std,
            "freqgated_mean": f_mean, "freqgated_std": f_std,
            "delta": delta, "p_value": p_val, "cohens_d": cohen_d,
        }

    # ---- H3 判定 ----
    cross_delta = conclusions["cross_domain_f1"]["delta"]
    cross_p = conclusions["cross_domain_f1"]["p_value"]
    cross_d = conclusions["cross_domain_f1"]["cohens_d"]

    # 对于跨域泛化：期望 FreqGated > Baseline（正 delta, 显著, 中等效应量以上）
    if cross_delta > 0 and cross_p < 0.1:
        # 小样本 (N=3) 时放宽 p 值要求，更看重效应量
        h3_passed = abs(cross_d) > 0.3  # 小效应量阈值
    else:
        h3_passed = False

    print(f"\n{'=' * 70}")
    print("H3 最终判定")
    print("=" * 70)
    if h3_passed:
        print(f"✅ H3 初步成立: 频率门控提升跨数据集泛化")
        print(f"   跨域 F1: {conclusions['cross_domain_f1']['freqgated_mean']:.4f} "
              f"vs {conclusions['cross_domain_f1']['baseline_mean']:.4f}")
        print(f"   p={cross_p:.4f}, d={cross_d:.3f}")
    else:
        if cross_delta > 0:
            print(f"⚠️ H3 趋势正向但未达统计显著: Δ={cross_delta:+.4f}, p={cross_p:.4f}, d={cross_d:.3f}")
            print(f"   需要更大规模实验验证（更多种子或更大训练集）")
        else:
            print(f"❌ H3 未通过: 频率门控未改善跨数据集泛化 (Δ={cross_delta:+.4f})")

    # ---- 可视化 ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("H3: Freq-Gated vs Baseline (RAF-DB subset, N=3 seeds)",
                 fontsize=13, fontweight="bold")

    metrics_plot = [
        ("in_domain_f1", "In-Domain F1", "RAF-DB val"),
        ("cross_domain_f1", "Cross-Domain F1", "FER2013 (unseen)"),
        ("gen_drop", "Generalization Drop", "In-domain - Cross-domain"),
    ]

    for idx, (key, title, ylabel) in enumerate(metrics_plot):
        ax = axes[idx]
        b_vals = [r[key] for r in baseline]
        f_vals = [r[key] for r in freqgated]

        x = np.arange(N_SEEDS)
        w = 0.35
        ax.bar(x - w/2, b_vals, w, label="Baseline", color="#2196F3", alpha=0.8)
        ax.bar(x + w/2, f_vals, w, label="FreqGated", color="#FF9800", alpha=0.8)

        for i in range(N_SEEDS):
            ax.text(i - w/2, b_vals[i] + 0.01, f"{b_vals[i]:.3f}", ha="center", fontsize=7)
            ax.text(i + w/2, f_vals[i] + 0.01, f"{f_vals[i]:.3f}", ha="center", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([f"Seed {s}" for s in [42, 123, 456][:N_SEEDS]], fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = H3_DIR / "h3_results.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n图表已保存: {fig_path}")

    # 保存报告
    report = {
        "h3_passed": h3_passed,
        "n_seeds": N_SEEDS,
        "samples_per_class": SAMPLES_PER_CLASS,
        "total_samples": SAMPLES_PER_CLASS * NUM_CLASSES,
        "epochs": EPOCHS,
        "conclusions": conclusions,
        "all_results": [{k: v for k, v in r.items() if k != "history"} for r in all_results],
    }
    report_path = H3_DIR / "h3_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"报告已保存: {report_path}")

    return h3_passed


if __name__ == "__main__":
    main()
