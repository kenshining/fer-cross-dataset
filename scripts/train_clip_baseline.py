"""
CLIP ViT-B/32 全量微调跨数据集泛化实验。

目的：回应 The Visual Computer 编辑关于 vision-language / foundation-model
      baseline 的要求。CLIP ViT-B/32 (OpenAI, 4亿图文对预训练) 代表视觉语言
      基础模型家族。

对比公平性说明：
  - 输入分辨率：224x224 (与 ResNet-18 一致)
  - 训练配置：40 epochs, Adam, lr=1e-3, ReduceLROnPlateau (与 ResNet-18 一致)
  - 参数量：~88M (CLIP ViT-B/32) vs ~11M (ResNet-18)
    此差异在 Limitations 中注明

用法:
  python fer_wavelet/scripts/train_clip_baseline.py --source rafdb
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

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

DATA_ROOT = Path("e:/scientific/小波/data")
RUNS_ROOT = _REPO / "runs" / "clip_baseline"
BATCH_SIZE = 16
NUM_CLASSES = 7
EPOCHS = 40
LR = 1e-3
SEEDS = [42, 123]

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

os.makedirs(RUNS_ROOT, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# Data Loaders
# ═══════════════════════════════════════════════════════════════════

def build_rafdb_train_loader(batch_size: int, input_size: int = 224):
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


def build_target_loader(dataset_name: str, batch_size: int, input_size: int = 224):
    from src.dataset_registry import REGISTRY
    from src.dataset_fer2013 import FER2013Dataset, fer2013_collate_fn

    if dataset_name == "rafdb":
        ds = REGISTRY["rafdb"]["dataset_cls"](DATA_ROOT / "RAF-DB", split="test")
        collate = None
    elif dataset_name == "fer2013":
        import csv
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
# Model
# ═══════════════════════════════════════════════════════════════════

class CLIPViTFER(nn.Module):
    """CLIP ViT-B/32 全量微调。

    Backbone: ViT-B/32，预训练于 OpenAI CLIP (4亿图文对)。
    参数量: ~88M (ViT-B/32) + 分类头。
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        from transformers import CLIPVisionModel
        self.vision = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        hidden_dim = self.vision.config.hidden_size  # 768
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out = self.vision(x, output_hidden_states=False)
        pooled = out.pooler_output  # [B, 768]
        return self.classifier(pooled)


# ═══════════════════════════════════════════════════════════════════
# Evaluate & Train
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

def run(source: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"CLIP ViT-B/32 Baseline: {source.upper()} source")
    print(f"Seeds: {SEEDS}  |  Epochs: {EPOCHS}  |  LR: {LR}")
    print(f"{'='*70}")

    if source == "rafdb":
        train_loader = build_rafdb_train_loader(BATCH_SIZE, 224)
        val_loader = build_target_loader("rafdb", BATCH_SIZE, 224)
        targets = ["fer2013", "affectnet", "ckplus", "jaffe"]
    elif source == "fer2013":
        from scripts.train_face_ablation import build_fer2013_train_loader
        train_loader = build_fer2013_train_loader(BATCH_SIZE, 224)
        val_loader = build_target_loader("fer2013", BATCH_SIZE, 224)
        targets = ["rafdb", "affectnet", "ckplus", "jaffe"]
    else:
        raise ValueError(f"Unknown source: {source}")

    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Targets: {targets}")

    target_loaders = {}
    for tgt in targets:
        try:
            target_loaders[tgt] = build_target_loader(tgt, BATCH_SIZE, 224)
            print(f"  {tgt}: {len(target_loaders[tgt].dataset)} test samples")
        except Exception as e:
            print(f"  [WARN] {tgt}: {e}")

    all_results = []

    for seed in SEEDS:
        tag = f"CLIP ViT-B/32 (seed={seed})"
        print(f"\n--- {tag} ---")
        model = CLIPViTFER()
        model, best_f1 = train_model(model, train_loader, val_loader, device, seed)
        print(f"  In-domain best Macro-F1: {best_f1:.4f}")
        model.eval()
        for tgt in targets:
            if tgt in target_loaders:
                m = evaluate(model, target_loaders[tgt], device)
                all_results.append({
                    "method": "CLIP-ViT-B/32", "source": source, "target": tgt,
                    "seed": seed,
                    "in_domain_f1": best_f1, "cross_domain_f1": m["macro_f1"],
                })
                print(f"    -> {tgt}: Macro-F1 = {m['macro_f1']:.4f}")
        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"SUMMARY: {source.upper()} source -- CLIP ViT-B/32")
    print(f"{'Target':<12} {'F1_mean':>10}")
    print("-" * 25)
    for tgt in targets:
        vals = [r["cross_domain_f1"] for r in all_results if r["target"] == tgt]
        print(f"{tgt:<12} {np.mean(vals):>10.4f}" if vals else f"{tgt:<12} {'N/A':>10}")

    out_path = RUNS_ROOT / f"{source}_clip_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="CLIP ViT-B/32 baseline experiment")
    parser.add_argument("--source", type=str, default="rafdb",
                        choices=["rafdb", "fer2013", "all"],
                        help="Source dataset (default: rafdb)")
    args = parser.parse_args()

    if args.source == "all":
        for src in ["rafdb", "fer2013"]:
            run(src)
    else:
        run(args.source)


if __name__ == "__main__":
    main()
