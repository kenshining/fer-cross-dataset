"""
人脸预训练消融实验：ImageNet vs MS1MV3 ArcFace 预训练 ResNet-18 跨数据集泛化对比。

目的：回应 The Visual Computer 编辑关于 MHAN 预训练域 confound 的关切。
     通过将 ResNet-18 的 ImageNet 预训练替换为 MS1MV3 ArcFace 人脸预训练，
     分离"预训练域"与"方法架构"对跨数据集泛化的贡献。

实验设计：
  - 源域: RAF-DB / FER2013（两个源都跑）
  - 目标域: FER2013/RAF-DB, AffectNet, CK+, JAFFE
  - 方法: ResNet-18 (ImageNet) vs ResNet-18 (MS1MV3 ArcFace)
  - 种子: 42, 123（各 2 seeds，与论文正文一致）
  - 训练配置: 与论文完全一致（40 epochs, Adam lr=1e-3, ReduceLROnPlateau）

用法:
  python fer_wavelet/scripts/train_face_ablation.py --source rafdb
  python fer_wavelet/scripts/train_face_ablation.py --source fer2013
  python fer_wavelet/scripts/train_face_ablation.py --source all
"""
from __future__ import annotations

import argparse, json, os, sys, tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path("e:/scientific/小波/MHAN-code/MHAN-main")))

DATA_ROOT = Path("e:/scientific/小波/data")
RUNS_ROOT = _REPO / "runs" / "face_ablation"
BATCH_SIZE = 16
NUM_CLASSES = 7
EPOCHS = 40
LR = 1e-3
SEEDS = [42, 123]

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ── 人脸预训练权重 ──
FACE_WEIGHTS_PATH = _REPO / "pretrained" / "ms1mv3_arcface_r18.pth"

os.makedirs(RUNS_ROOT, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# Data Loaders
# ═══════════════════════════════════════════════════════════════════

def build_rafdb_train_loader(batch_size: int, input_size: int = 224):
    """RAF-DB 训练集 DataLoader。

    RAF-DB 原生 collate_fn 返回 (list[PIL], labels)，需要改为返回 tensor。
    这里用 Wrapper 预转为 tensor，使用默认 collate_fn。
    """
    from src.dataset_registry import REGISTRY
    ds = REGISTRY["rafdb"]["dataset_cls"](DATA_ROOT / "RAF-DB", split="train")
    t = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    class Wrapper(torch.utils.data.Dataset):
        def __init__(self, ds, t):
            self.ds = ds; self.t = t
        def __len__(self): return len(self.ds)
        def __getitem__(self, idx):
            pil, label = self.ds[idx]
            if not isinstance(pil, Image.Image):
                pil = pil.convert("RGB") if hasattr(pil, "convert") else Image.fromarray(np.array(pil))
            return self.t(pil), label
    return DataLoader(Wrapper(ds, t), batch_size=batch_size, shuffle=True, num_workers=0)


def build_fer2013_train_loader(batch_size: int, input_size: int = 224):
    """FER2013 训练集 DataLoader（从 CSV 中提取 Usage=Training 行）。"""
    import csv
    csv_path = DATA_ROOT / "Fer2013" / "fer2013" / "fer2013.csv"

    class FER2013TensorDS(torch.utils.data.Dataset):
        def __init__(self):
            self.rows = []
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("Usage", "") == "Training":
                        self.rows.append((row["pixels"], int(row["emotion"])))
        def __len__(self): return len(self.rows)
        def __getitem__(self, idx):
            pixels_str, label = self.rows[idx]
            pix = np.fromstring(pixels_str, sep=" ", dtype=np.uint8)
            img = pix.reshape(48, 48)
            img = np.stack([img]*3, axis=-1)
            pil = Image.fromarray(img).resize((input_size, input_size), Image.BILINEAR)
            tensor = transforms.ToTensor()(pil)
            tensor = transforms.Normalize(mean=MEAN, std=STD)(tensor)
            return tensor, label

    ds = FER2013TensorDS()
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)


def build_target_loader(dataset_name: str, batch_size: int, input_size: int = 224):
    """构建目标域 DataLoader（复用 train_cross_source.py 的逻辑）。"""
    from src.dataset_registry import REGISTRY
    from src.dataset_fer2013 import FER2013Dataset, fer2013_collate_fn

    if dataset_name == "rafdb":
        ds = REGISTRY["rafdb"]["dataset_cls"](DATA_ROOT / "RAF-DB", split="test")
        collate = None  # 使用默认 collate（Wrapper 已预转为 tensor）
    elif dataset_name == "fer2013":
        csv_path = DATA_ROOT / "Fer2013" / "fer2013" / "fer2013.csv"
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8")
        tmp.write("emotion,pixels\n")
        with open(csv_path) as fin:
            next(fin)
            for line in fin:
                if "PublicTest" in line:
                    parts = line.strip().split(",", 2)
                    if len(parts) >= 2:
                        tmp.write(f"{parts[0]},{parts[1]}\n")
        tmp.close()
        ds = FER2013Dataset(Path(tmp.name))
        collate = fer2013_collate_fn
    elif dataset_name == "affectnet":
        ds = REGISTRY["affectnet"]["dataset_cls"](DATA_ROOT / "AffectNet", split="val")
        collate = REGISTRY["affectnet"]["collate_fn"]
    elif dataset_name == "ckplus":
        ds = REGISTRY["ckplus"]["dataset_cls"](DATA_ROOT / "CK+")
        collate = REGISTRY["ckplus"]["collate_fn"]
    elif dataset_name == "jaffe":
        ds = REGISTRY["jaffe"]["dataset_cls"](DATA_ROOT / "Jaffe")
        collate = REGISTRY["jaffe"]["collate_fn"]
    else:
        raise ValueError(f"Unknown target: {dataset_name}")

    t = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    class Wrapper(torch.utils.data.Dataset):
        def __init__(self, ds, t):
            self.ds = ds; self.t = t
        def __len__(self): return len(self.ds)
        def __getitem__(self, idx):
            pil, label = self.ds[idx]
            if not isinstance(pil, Image.Image):
                pil = pil.convert("RGB") if hasattr(pil, "convert") else Image.fromarray(np.array(pil))
            return self.t(pil), label

    return DataLoader(Wrapper(ds, t), batch_size=batch_size, shuffle=False, num_workers=0)


# ═══════════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════════

class ResNetFER(nn.Module):
    """标准 ResNet-18 + ImageNet 预训练（与论文正文完全一致）。"""
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(512, 256),
                                        nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, NUM_CLASSES))
    def forward(self, x):
        return self.classifier(self.encoder(x))


class ResNetFER_Face(nn.Module):
    """iresnet18 + MS1MV3 ArcFace 人脸预训练（消融实验）。"""
    def __init__(self, face_weights_path: str):
        super().__init__()
        from src.models import build_iresnet18_face_backbone
        self.encoder, _ = build_iresnet18_face_backbone(face_weights_path)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(512, 256),
                                        nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, NUM_CLASSES))
    def forward(self, x):
        return self.classifier(self.encoder(x))


# ═══════════════════════════════════════════════════════════════════
# Evaluate
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0; total = 0
    tp = torch.zeros(NUM_CLASSES, device=device)
    fp = torch.zeros(NUM_CLASSES, device=device)
    fn = torch.zeros(NUM_CLASSES, device=device)
    for batch_data, labels in loader:
        rgb, labels = batch_data.to(device), labels.to(device)
        out = model(rgb)
        if isinstance(out, tuple):
            out = out[0]
        pred = out.argmax(dim=1)
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
    return {"acc": acc, "macro_f1": f1.mean().item()}


def train_model(model, train_loader, val_loader, device, seed: int):
    """标准训练循环（与论文正文完全一致）。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
    crit = nn.CrossEntropyLoss()
    best_f1, best_state = 0.0, None
    for ep in range(1, EPOCHS + 1):
        model.train()
        for batch_data, labels in train_loader:
            rgb, labels = batch_data.to(device), labels.to(device)
            opt.zero_grad()
            loss = crit(model(rgb), labels)
            loss.backward(); opt.step()
        m = evaluate(model, val_loader, device)
        sch.step(m["macro_f1"])
        if m["macro_f1"] > best_f1:
            best_f1, best_state = m["macro_f1"], deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best_f1


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def run_ablation(source: str):
    """对指定 source 运行 ImageNet vs Face pretrain 消融实验。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"Face Pretrain Ablation: {source.upper()} source")
    print(f"Comparing: ResNet-18 (ImageNet) vs ResNet-18 (MS1MV3 ArcFace)")
    print(f"Seeds: {SEEDS}  |  Epochs: {EPOCHS}  |  LR: {LR}")
    print(f"{'='*70}")

    # ── 构建训练和验证 DataLoader ──
    if source == "rafdb":
        train_loader = build_rafdb_train_loader(BATCH_SIZE, 224)
        val_loader = build_target_loader("rafdb", BATCH_SIZE, 224)
        targets = ["fer2013", "affectnet", "ckplus", "jaffe"]
    elif source == "fer2013":
        train_loader = build_fer2013_train_loader(BATCH_SIZE, 224)
        val_loader = build_target_loader("fer2013", BATCH_SIZE, 224)
        targets = ["rafdb", "affectnet", "ckplus", "jaffe"]
    else:
        raise ValueError(f"Unknown source: {source}")

    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Targets: {targets}")

    # ── 构建目标域 DataLoader ──
    target_loaders = {}
    for tgt in targets:
        try:
            target_loaders[tgt] = build_target_loader(tgt, BATCH_SIZE, 224)
            print(f"  {tgt}: {len(target_loaders[tgt].dataset)} test samples")
        except Exception as e:
            print(f"  [WARN] {tgt}: {e}")

    all_results = []

    # ── 实验 A: ResNet-18 + ImageNet pretrain ──
    for seed in SEEDS:
        tag = f"ResNet-ImageNet (seed={seed})"
        print(f"\n--- {tag} ---")
        model = ResNetFER()
        model, best_f1 = train_model(model, train_loader, val_loader, device, seed)
        print(f"  In-domain best Macro-F1: {best_f1:.4f}")
        model.eval()
        for tgt in targets:
            if tgt in target_loaders:
                m = evaluate(model, target_loaders[tgt], device)
                all_results.append({
                    "pretrain": "ImageNet", "method": "ResNet",
                    "source": source, "target": tgt, "seed": seed,
                    "in_domain_f1": best_f1, "cross_domain_f1": m["macro_f1"],
                })
                print(f"    → {tgt}: Macro-F1 = {m['macro_f1']:.4f}")
        torch.cuda.empty_cache()

    # ── 实验 B: ResNet-18 + MS1MV3 ArcFace pretrain ──
    if not FACE_WEIGHTS_PATH.exists():
        print(f"\n[SKIP] Face weights not found: {FACE_WEIGHTS_PATH}")
        print("  Download from: https://1drv.ms/u/s!AswpsDO2toNKq0lWY69vN58GR6mw?e=p9Ov5d")
        print("  Place at: fer_wavelet/pretrained/ms1mv3_arcface_r18.pth")
    else:
        for seed in SEEDS:
            tag = f"ResNet-Face (seed={seed})"
            print(f"\n--- {tag} ---")
            model = ResNetFER_Face(str(FACE_WEIGHTS_PATH))
            model, best_f1 = train_model(model, train_loader, val_loader, device, seed)
            print(f"  In-domain best Macro-F1: {best_f1:.4f}")
            model.eval()
            for tgt in targets:
                if tgt in target_loaders:
                    m = evaluate(model, target_loaders[tgt], device)
                    all_results.append({
                        "pretrain": "MS1MV3_ArcFace", "method": "ResNet",
                        "source": source, "target": tgt, "seed": seed,
                        "in_domain_f1": best_f1, "cross_domain_f1": m["macro_f1"],
                    })
                    print(f"    → {tgt}: Macro-F1 = {m['macro_f1']:.4f}")
            torch.cuda.empty_cache()

    # ── 汇总 ──
    print(f"\n{'='*70}")
    print(f"SUMMARY: {source.upper()} source — ImageNet vs Face Pretrain")
    print(f"{'Target':<12} {'ImageNet':>10} {'Face':>10} {'Δ':>10}")
    print("-" * 46)
    for tgt in targets:
        inet_vals = [r["cross_domain_f1"] for r in all_results
                     if r["target"] == tgt and r["pretrain"] == "ImageNet"]
        face_vals = [r["cross_domain_f1"] for r in all_results
                     if r["target"] == tgt and r["pretrain"] == "MS1MV3_ArcFace"]
        inet_mean = np.mean(inet_vals) if inet_vals else 0
        face_mean = np.mean(face_vals) if face_vals else 0
        delta = face_mean - inet_mean
        print(f"{tgt:<12} {inet_mean:>10.4f} {face_mean:>10.4f} {delta:>+10.4f}")

    # ── 保存结果 ──
    out_path = RUNS_ROOT / f"{source}_face_ablation.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Face pretrain ablation experiment")
    parser.add_argument("--source", type=str, default="rafdb",
                        choices=["rafdb", "fer2013", "all"],
                        help="Source dataset (default: rafdb)")
    args = parser.parse_args()

    if args.source == "all":
        for src in ["rafdb", "fer2013"]:
            run_ablation(src)
    else:
        run_ablation(args.source)


if __name__ == "__main__":
    main()
