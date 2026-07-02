"""
t-SNE 特征可视化: 对比不同方法在域内 vs 跨域的特征分布
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
from sklearn.manifold import TSNE
import tempfile

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path("e:/scientific/小波/MHAN-code/MHAN-main")))

from src.scn_model import SelfAttentionWeighting

OUTPUT_DIR = _REPO / "paper" / "figures"
DATA_ROOT = Path("e:/scientific/小波/data")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 7

os.makedirs(OUTPUT_DIR, exist_ok=True)


class ResNetFE(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
    def forward(self, x):
        return self.encoder(x).view(x.size(0), -1)


class SCNFE(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])  # avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        f = self.encoder(x)
        return self.avgpool(f).view(x.size(0), -1)


def get_mhan_features(model, loader):
    """Extract MHAN features (post-ELA, pre-head)."""
    model.eval()
    feats, lbls = [], []
    with torch.no_grad():
        for rgb, labels in loader:
            rgb = rgb.to(DEVICE)
            _, feat, _ = model(rgb)
            # feat is spatial (B, C, H, W) — pool to (B, C)
            f = feat.mean(dim=[2, 3]).cpu().numpy()
            feats.append(f)
            lbls.append(labels.numpy())
    return np.concatenate(feats), np.concatenate(lbls)


def get_resnet_features(model, loader):
    model.eval()
    feats, lbls = [], []
    with torch.no_grad():
        for rgb, labels in loader:
            rgb = rgb.to(DEVICE)
            f = model(rgb).cpu().numpy()
            feats.append(f)
            lbls.append(labels.numpy())
    return np.concatenate(feats), np.concatenate(lbls)


def build_fer_loader(input_size: int = 224, n_samples: int = 200):
    """FER2013 test loader."""
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
    rows = rows[:n_samples]

    tensors, labels = [], []
    for pixels_str, label in rows:
        pix = np.fromstring(pixels_str, sep=" ", dtype=np.uint8)
        img = pix.reshape(48, 48)
        img = np.stack([img]*3, axis=-1)
        pil = Image.fromarray(img).resize((input_size, input_size), Image.BILINEAR)
        t = transforms.ToTensor()(pil)
        t = transforms.Normalize(mean=MEAN, std=STD)(t)
        tensors.append(t)
        labels.append(label)

    ds = torch.utils.data.TensorDataset(torch.stack(tensors), torch.tensor(labels))
    return DataLoader(ds, batch_size=64, shuffle=False)


def build_raf_loader(input_size: int = 224, n_samples: int = 200):
    """RAF-DB test loader."""
    from src.dataset_registry import REGISTRY
    ds = REGISTRY["rafdb"]["dataset_cls"](DATA_ROOT / "RAF-DB", split="test")

    indices = np.random.choice(len(ds), min(n_samples, len(ds)), replace=False)
    t = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    class Wrapper(torch.utils.data.Dataset):
        def __init__(self, ds, indices, t):
            self.ds = ds; self.indices = indices; self.t = t
        def __len__(self): return len(self.indices)
        def __getitem__(self, idx):
            pil, label = self.ds[self.indices[idx]]
            return self.t(pil.convert("RGB")), label

    return DataLoader(Wrapper(ds, indices, t), batch_size=64, shuffle=False)


def main():
    print("Loading data...")
    fer_loader_224 = build_fer_loader(224, 200)
    fer_loader_112 = build_fer_loader(112, 200)
    raf_loader_224 = build_raf_loader(224, 200)
    raf_loader_112 = build_raf_loader(112, 200)

    emotion_names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    # Load trained models
    runs = _REPO / "runs"

    # ResNet (RAF-DB trained)
    print("Extracting ResNet features...")
    resnet = ResNetFE().to(DEVICE)
    ckpt = torch.load(runs / "rafdb_rgb" / "best.pt", map_location=DEVICE, weights_only=False)
    resnet.load_state_dict({k.replace("encoder.", ""): v for k, v in ckpt["model"].items()
                            if "encoder" in k}, strict=False)
    resnet.eval()

    raf_rn = get_resnet_features(resnet, raf_loader_224)
    fer_rn = get_resnet_features(resnet, fer_loader_224)

    # SCN (RAF-DB trained)
    print("Extracting SCN features...")
    scn = SCNFE().to(DEVICE)
    ckpt_scn = torch.load(runs / "scn_baseline" / "scn_rafdb_seed42" / "best.pt",
                          map_location=DEVICE, weights_only=False)
    # Load encoder part
    scn_state = {k.replace("encoder.", ""): v for k, v in ckpt_scn["model"].items()
                 if "encoder" in k and "classifier" not in k}
    scn.load_state_dict(scn_state, strict=False)
    scn.eval()

    raf_scn = get_resnet_features(scn, raf_loader_224)
    fer_scn = get_resnet_features(scn, fer_loader_224)

    # MHAN (RAF-DB trained)
    print("Extracting MHAN features...")
    from networks.backbone import MHAN
    mhan = MHAN(num_class=NUM_CLASSES, num_head=2, pretrained=False).to(DEVICE)
    ckpt_m = torch.load(runs / "mhan_baseline" / "mhan_rafdb_seed42.pt",
                        map_location=DEVICE, weights_only=False)
    mhan.load_state_dict(ckpt_m["model"])
    mhan.eval()

    raf_mhan = get_mhan_features(mhan, raf_loader_112)
    fer_mhan = get_mhan_features(mhan, fer_loader_112)

    # ---- t-SNE ----
    print("Computing t-SNE...")
    # Combine all features for joint embedding
    all_feats = np.concatenate([
        raf_rn[0], fer_rn[0],
        raf_scn[0], fer_scn[0],
        raf_mhan[0], fer_mhan[0],
    ])
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    all_2d = tsne.fit_transform(all_feats)

    n_per_fer = len(fer_rn[0])
    n_per_raf = len(raf_rn[0])
    idx = 0
    rn_raf_2d = all_2d[idx:idx + n_per_raf]; idx += n_per_raf
    rn_fer_2d = all_2d[idx:idx + n_per_fer]; idx += n_per_fer
    scn_raf_2d = all_2d[idx:idx + n_per_raf]; idx += n_per_raf
    scn_fer_2d = all_2d[idx:idx + n_per_fer]; idx += n_per_fer
    mhan_raf_2d = all_2d[idx:idx + n_per_raf]; idx += n_per_raf
    mhan_fer_2d = all_2d[idx:idx + n_per_fer]

    # ---- Plot: 3×2 layout (Method × Domain) + right-side shared legend ----
    fig, axes = plt.subplots(3, 2, figsize=(18.48, 23.76))
    emotion_hex = ["#154760", "#2c6e85", "#5a8fa3", "#8ab0bf", "#c99595", "#c46b6b", "#bf1a24"]
    colors = [plt.matplotlib.colors.to_rgb(h) for h in emotion_hex]

    method_names = ["ResNet-18", "SCN", "MHAN"]
    datasets_2d = [
        [(rn_raf_2d, raf_rn[1], "In-Domain (RAF-DB)"),
         (rn_fer_2d, fer_rn[1], "Cross-Domain (FER2013)")],
        [(scn_raf_2d, raf_scn[1], "In-Domain (RAF-DB)"),
         (scn_fer_2d, fer_scn[1], "Cross-Domain (FER2013)")],
        [(mhan_raf_2d, raf_mhan[1], "In-Domain (RAF-DB)"),
         (mhan_fer_2d, fer_mhan[1], "Cross-Domain (FER2013)")],
    ]

    for row_idx, (method_name, row_data) in enumerate(zip(method_names, datasets_2d)):
        for col_idx, (xy, labels, domain_label) in enumerate(row_data):
            ax = axes[row_idx, col_idx]
            for c in range(7):
                mask = labels == c
                ax.scatter(xy[mask, 0], xy[mask, 1], c=[colors[c]], s=90,
                          alpha=0.55, edgecolors="none")
            ax.set_title(f"{method_name}: {domain_label}", fontsize=16, fontweight="bold")
            ax.set_xticklabels([]); ax.set_yticklabels([])
            ax.set_axisbelow(True)
            ax.grid(True, linestyle="--", linewidth=0.6, color="#aaaaaa", alpha=0.35)

            if col_idx == 0:
                ax.set_ylabel(method_name, fontsize=16, fontweight="bold", rotation=90, labelpad=8)

    # Bottom horizontal legend — "Emotions" inline with items on same row
    from matplotlib.lines import Line2D
    dummy = Line2D([0], [0], linestyle="none", marker="", label="Emotions")
    handles_emo = [dummy] + [Line2D([0], [0], marker="o", linestyle="none",
                           markerfacecolor=colors[i], markersize=14, label=emotion_names[i])
                            for i in range(7)]
    leg = fig.legend(handles=handles_emo, loc="upper center", ncol=8, fontsize=16,
                     frameon=True, columnspacing=0.8, handletextpad=0.6, handlelength=0.8,
                     bbox_to_anchor=(0.5, 0.065), borderaxespad=0)

    plt.subplots_adjust(left=0.06, right=0.97, bottom=0.09, top=0.94, wspace=0.08, hspace=0.12)
    for fmt in ["png", "svg", "eps"]:
        fig.savefig(OUTPUT_DIR / f"fig7_tsne.{fmt}", dpi=600, bbox_inches="tight")
    plt.close()
    print("Fig 5: t-SNE saved")

    print("All analysis figures complete!")


if __name__ == "__main__":
    main()
