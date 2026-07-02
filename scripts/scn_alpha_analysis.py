"""
Task 3: SCN α 权重跨域行为分析

展示 SCN 的自愈机制如何感知域差异。
"""
from __future__ import annotations

import os, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image
from scipy import stats

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.scn_model import SelfAttentionWeighting

DATA_ROOT = Path("e:/scientific/小波/data")
RUNS = _REPO / "runs"
OUT = _REPO / "paper" / "figures"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 7
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

os.makedirs(OUT, exist_ok=True)


class SCNExtractor(nn.Module):
    """SCN 模型的特征 + α + logits 提取器."""
    def __init__(self, ckpt_path: str):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.alpha_module = SelfAttentionWeighting(512)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(256, NUM_CLASSES),
        )

        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model_state = ckpt["model"]
        enc_state = {k.replace("encoder.", ""): v for k, v in model_state.items() if k.startswith("encoder.")}
        self.encoder.load_state_dict(enc_state, strict=False)
        alpha_state = {k.replace("alpha_module.", ""): v for k, v in model_state.items() if k.startswith("alpha_module.")}
        self.alpha_module.load_state_dict(alpha_state, strict=True)
        cls_state = {k.replace("classifier.", ""): v for k, v in model_state.items() if k.startswith("classifier.")}
        if cls_state:
            self.classifier.load_state_dict(cls_state, strict=True)
        self.eval().to(DEVICE)

    @torch.no_grad()
    def extract(self, loader) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        alphas, labels_all, confs_all, preds_all = [], [], [], []
        for batch_data, labels in loader:
            rgb = batch_data.to(DEVICE)
            rgb = transforms.Normalize(mean=MEAN, std=STD)(rgb)
            f = self.encoder(rgb).view(rgb.size(0), -1)
            a = self.alpha_module(f)
            logits = self.classifier(f)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            alphas.append(a.cpu().numpy())
            labels_all.append(labels.numpy() if isinstance(labels, np.ndarray) else labels.cpu().numpy())
            confs_all.append(conf.cpu().numpy())
            preds_all.append(pred.cpu().numpy())
        return (np.concatenate(alphas), np.concatenate(labels_all),
                np.concatenate(confs_all), np.concatenate(preds_all))


def build_raf_loader(split="test", n=None):
    from src.dataset_registry import REGISTRY
    ds = REGISTRY["rafdb"]["dataset_cls"](DATA_ROOT / "RAF-DB", split=split)
    indices = list(range(len(ds)))
    if n:
        indices = np.random.choice(indices, min(n, len(ds)), replace=False)

    class W(torch.utils.data.Dataset):
        def __init__(s): s.ds = ds; s.idx = indices
        def __len__(s): return len(s.idx)
        def __getitem__(s, i):
            p, l = s.ds[s.idx[i]]
            p = p.convert("RGB") if hasattr(p, "convert") else p
            t = transforms.ToTensor()(transforms.Resize((224,224))(p))
            return t, l

    return DataLoader(W(), batch_size=64, shuffle=False)


def build_fer_loader(n=500):
    import csv
    csv_path = DATA_ROOT / "Fer2013" / "fer2013" / "fer2013.csv"
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Usage", "") == "PublicTest":
                rows.append((row["pixels"], int(row["emotion"])))
    np.random.seed(42)
    np.random.shuffle(rows)
    rows = rows[:n]

    tensors, labels = [], []
    for pixels_str, label in rows:
        pix = np.fromstring(pixels_str, sep=" ", dtype=np.uint8)
        img = pix.reshape(48, 48)
        img = np.stack([img]*3, axis=-1)
        pil = Image.fromarray(img).resize((224, 224), Image.BILINEAR)
        t = transforms.ToTensor()(pil)
        tensors.append(t)
        labels.append(label)
    return DataLoader(
        torch.utils.data.TensorDataset(torch.stack(tensors), torch.tensor(labels)),
        batch_size=64, shuffle=False,
    )


EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def main():
    ckpt = RUNS / "scn_baseline" / "scn_rafdb_seed42" / "best.pt"
    print(f"Loading SCN from: {ckpt}")
    model = SCNExtractor(str(ckpt))

    # Extract α for in-domain (RAF-DB) and cross-domain (FER2013)
    print("Extracting in-domain α (RAF-DB test)...")
    raf_loader = build_raf_loader("test")
    raf_a, raf_l, raf_conf, raf_pred = model.extract(raf_loader)

    print("Extracting cross-domain α (FER2013 test)...")
    fer_loader = build_fer_loader(500)
    fer_a, fer_l, fer_conf, fer_pred = model.extract(fer_loader)

    # ---- Figure 7: α distribution comparison ----
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))

    # (a) Overall α distribution
    ax = axes[0, 0]
    ax.hist(raf_a, bins=30, alpha=0.6, label=f"RAF-DB (in-domain, μ={raf_a.mean():.3f})", color="#154760")
    ax.hist(fer_a, bins=30, alpha=0.6, label=f"FER2013 (cross-domain, μ={fer_a.mean():.3f})", color="#bf1a24")
    ax.set_xlabel("SCN α (Importance Weight)", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.set_title("(a) α Distribution: In-Domain vs Cross-Domain", fontsize=16, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    # KS test below legend
    ks_s, ks_p = stats.ks_2samp(raf_a, fer_a)
    ax.text(0.98, 0.72, f"KS p={ks_p:.2e}", transform=ax.transAxes,
            ha="right", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="gray", linewidth=0.5))

    # (i) Bottom-right: α vs Prediction Confidence
    ax = axes[2, 2]
    raf_correct = (raf_pred == raf_l)
    fer_correct = (fer_pred == fer_l)
    ax.scatter(raf_a[raf_correct], raf_conf[raf_correct], c="#2196F3", s=8,
              alpha=0.5, label="RAF correct")
    ax.scatter(raf_a[~raf_correct], raf_conf[~raf_correct], c="#2196F3", s=8,
              alpha=0.2, marker="x", label="RAF wrong")
    ax.scatter(fer_a[fer_correct], fer_conf[fer_correct], c="#F44336", s=8,
              alpha=0.5, label="FER correct")
    ax.scatter(fer_a[~fer_correct], fer_conf[~fer_correct], c="#F44336", s=8,
              alpha=0.2, marker="x", label="FER wrong")

    # Correlation annotation above legend
    fer_r, fer_p = stats.pearsonr(fer_a, fer_conf)
    raf_r, raf_p = stats.pearsonr(raf_a, raf_conf)
    ax.text(0.98, 0.22, f"RAF r={raf_r:.3f}, p={raf_p:.2e}\nFER r={fer_r:.3f}, p={fer_p:.2e}",
            transform=ax.transAxes, fontsize=8, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="gray", linewidth=0.5))
    ax.set_xlabel("SCN α", fontsize=14)
    ax.set_ylabel("Max Prediction Probability", fontsize=14)
    ax.set_title("(i) α vs Model Confidence", fontsize=16, fontweight="bold")
    ax.legend(fontsize=6, loc="lower right", markerscale=1.5)
    ax.grid(alpha=0.3)

    # (b-h) Per-class α comparison (7 classes in remaining 8 slots)
    per_class_positions = [(0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1)]
    for c in range(7):
        r, col = per_class_positions[c]
        ax = axes[r, col]
        raf_c = raf_a[raf_l == c]
        fer_c = fer_a[fer_l == c]
        if len(raf_c) > 0:
            ax.hist(raf_c, bins=15, alpha=0.6, label=f"RAF (n={len(raf_c)}, μ={raf_c.mean():.3f})", color="#154760")
        if len(fer_c) > 0:
            ax.hist(fer_c, bins=15, alpha=0.6, label=f"FER (n={len(fer_c)}, μ={fer_c.mean():.3f})", color="#bf1a24")
        ax.set_title(EMOTIONS[c], fontsize=16, fontweight="bold")
        ax.set_xlabel("α", fontsize=14); ax.set_xlim(0.4, 0.85)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    for fmt in ["png", "svg", "eps"]:
        fig.savefig(OUT / f"fig5_scn_alpha.{fmt}", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Fig 7 saved: {OUT / 'fig5_scn_alpha.png'}")

    # ---- Per-class statistics ----
    print(f"\nSCN α Per-Class Analysis:")
    print(f"{'Emotion':<12} {'RAF μ(α)':>10} {'FER μ(α)':>10} {'Δ':>10} {'Δ%':>10}")
    print("-" * 55)
    for c in range(7):
        raf_mu = raf_a[raf_l == c].mean() if sum(raf_l == c) > 0 else 0
        fer_mu = fer_a[fer_l == c].mean() if sum(fer_l == c) > 0 else 0
        delta = fer_mu - raf_mu
        print(f"{EMOTIONS[c]:<12} {raf_mu:>10.4f} {fer_mu:>10.4f} {delta:>+10.4f} {delta/raf_mu*100:>+9.1f}%")

    print(f"\nOverall: RAF μ(α)={raf_a.mean():.4f}, FER μ(α)={fer_a.mean():.4f}, "
          f"Δ={fer_a.mean()-raf_a.mean():+.4f}")

    if fer_a.mean() < raf_a.mean():
        print("✅ SCN correctly perceives cross-domain data as more uncertain (lower α).")
    else:
        print("⚠️ SCN α did not decrease on cross-domain data.")

    print("Done!")


if __name__ == "__main__":
    main()
