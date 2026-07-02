"""
ViT 基线训练 + 跨数据集评估

ViT-B/16 (Vision Transformer) — 代表 Transformer 架构家族
与 ResNet-18 (CNN), SCN (标签噪声), RUL (不确定性) 形成架构与方法论两个维度的对比.
"""
from __future__ import annotations

import json, os, sys, tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import vit_b_16, ViT_B_16_Weights

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.preprocess import center_crop_resize, pil_to_tensor01
from src.dataset_registry import REGISTRY

DATA_ROOT = Path("e:/scientific/小波/data")
RUNS_ROOT = _REPO / "runs" / "vit_baseline"
BATCH_SIZE = 16
FACE_SIZE = 224
NUM_CLASSES = 7
EPOCHS = 40
LR = 1e-4  # ViT needs lower LR
SEEDS = [42, 123]

os.makedirs(RUNS_ROOT, exist_ok=True)


class ViTFER(nn.Module):
    """ViT-B/16 for FER, ImageNet-21k pretrained."""

    def __init__(self, num_classes: int = 7):
        super().__init__()
        weights = ViT_B_16_Weights.DEFAULT
        self.vit = vit_b_16(weights=weights)
        # Replace head
        self.vit.heads = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor):
        return self.vit(x)


def build_loader(dataset_name: str, split: str, batch_size: int, shuffle: bool):
    if dataset_name == "rafdb":
        ds = REGISTRY["rafdb"]["dataset_cls"](DATA_ROOT / "RAF-DB", split=split)
        collate = REGISTRY["rafdb"]["collate_fn"]
    elif dataset_name == "fer2013":
        from src.dataset_fer2013 import FER2013Dataset, fer2013_collate_fn
        csv_path = DATA_ROOT / "Fer2013" / "fer2013" / "fer2013.csv"
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8")
        tmp.write("emotion,pixels\n")
        with open(csv_path) as fin:
            for line in fin:
                if "PublicTest" in line:
                    parts = line.strip().split(",", 2)
                    if len(parts) >= 2:
                        tmp.write(f"{parts[0]},{parts[1]}\n")
        tmp.close()
        ds = FER2013Dataset(Path(tmp.name))
        collate = fer2013_collate_fn
    elif dataset_name == "affectnet":
        ds = REGISTRY["affectnet"]["dataset_cls"](DATA_ROOT / "AffectNet", split=split)
        collate = REGISTRY["affectnet"]["collate_fn"]
    elif dataset_name == "ckplus":
        ds = REGISTRY["ckplus"]["dataset_cls"](DATA_ROOT / "CK+")
        collate = REGISTRY["ckplus"]["collate_fn"]
    elif dataset_name == "jaffe":
        ds = REGISTRY["jaffe"]["dataset_cls"](DATA_ROOT / "Jaffe")
        collate = REGISTRY["jaffe"]["collate_fn"]
    else:
        raise ValueError(f"Unknown: {dataset_name}")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0; total = 0
    tp = torch.zeros(NUM_CLASSES, device=device)
    fp = torch.zeros(NUM_CLASSES, device=device)
    fn = torch.zeros(NUM_CLASSES, device=device)

    for batch_data, labels in loader:
        labels = labels.to(device)
        if isinstance(batch_data, torch.Tensor):
            rgb = batch_data.to(device)
        else:
            rgb_list = [pil_to_tensor01(center_crop_resize(p.convert("RGB"), FACE_SIZE)) for p in batch_data]
            rgb = torch.stack(rgb_list).to(device)
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
    return {"acc": acc, "macro_f1": f1.mean().item()}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}\nViT-B/16 基线: RAF-DB → 跨数据集 (2 seeds)")

    train_loader = build_loader("rafdb", "train", BATCH_SIZE, shuffle=True)
    val_loader = build_loader("rafdb", "test", BATCH_SIZE, shuffle=False)
    print(f"RAF-DB: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}")

    target_names = ["fer2013", "affectnet", "ckplus", "jaffe"]
    target_loaders = {}
    for tgt in target_names:
        try:
            split = "val" if tgt == "affectnet" else "test"
            target_loaders[tgt] = build_loader(tgt, split, BATCH_SIZE, shuffle=False)
            print(f"  {tgt}: {len(target_loaders[tgt].dataset)} 样本")
        except Exception as e:
            print(f"  [WARN] {tgt}: {e}")

    all_results = []
    criterion = nn.CrossEntropyLoss()

    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        print(f"\n{'='*50}\nSeed {seed}\n{'='*50}")

        model = ViTFER(num_classes=NUM_CLASSES).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_val_f1 = 0.0
        best_state = None

        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss = 0.0; correct = 0; total_s = 0

            for batch_data, labels in train_loader:
                labels = labels.to(device)
                if isinstance(batch_data, torch.Tensor):
                    rgb = batch_data.to(device)
                else:
                    rgb_list = [pil_to_tensor01(center_crop_resize(p.convert("RGB"), FACE_SIZE)) for p in batch_data]
                    rgb = torch.stack(rgb_list).to(device)

                optimizer.zero_grad()
                logits = model(rgb)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * labels.size(0)
                pred = logits.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total_s += labels.size(0)

            scheduler.step()
            val_m = evaluate(model, val_loader, device)

            if val_m["macro_f1"] > best_val_f1:
                best_val_f1 = val_m["macro_f1"]
                best_state = deepcopy(model.state_dict())

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:2d}: loss={total_loss/total_s:.4f}, "
                      f"val_f1={val_m['macro_f1']:.4f}", flush=True)

        model.load_state_dict(best_state)
        torch.save({"model": best_state, "best_val_f1": best_val_f1},
                   RUNS_ROOT / f"vit_rafdb_seed{seed}.pt")
        model.eval()
        print(f"  训练完成: best_val_f1={best_val_f1:.4f}")

        for tgt_name, tgt_loader in target_loaders.items():
            metrics = evaluate(model, tgt_loader, device)
            all_results.append({"seed": seed, "source": "rafdb", "target": tgt_name,
                                "acc": metrics["acc"], "macro_f1": metrics["macro_f1"]})
            print(f"    → {tgt_name}: f1={metrics['macro_f1']:.4f}")
        torch.cuda.empty_cache()

    # 汇总
    baseline_f1s = {"fer2013": 0.2969, "affectnet": 0.2491, "ckplus": 0.1739, "jaffe": 0.1534}
    scn_ref = {"fer2013": 0.3666, "affectnet": 0.3259, "ckplus": 0.2250, "jaffe": 0.1842}
    rul_ref = {"fer2013": 0.3677, "affectnet": 0.3250, "ckplus": 0.2271, "jaffe": 0.1384}

    print(f"\n{'='*60}")
    print(f"ResNet-18 | SCN | RUL | ViT 对比 (RAF-DB → X)")
    print(f"{'Target':<12} {'ResNet':>8} {'SCN':>8} {'RUL':>8} {'ViT':>8}")
    print("-" * 50)
    for tgt in target_names:
        f1s = [r["macro_f1"] for r in all_results if r["target"] == tgt]
        if f1s and not np.isnan(f1s[0]):
            m = np.mean(f1s)
            print(f"{tgt:<12} {baseline_f1s[tgt]:>8.4f} {scn_ref[tgt]:>8.4f} "
                  f"{rul_ref[tgt]:>8.4f} {m:>8.4f}")

    overall = np.mean([r["macro_f1"] for r in all_results if not np.isnan(r["macro_f1"])])
    print(f"\n{'平均':<12} {0.2183:>8.4f} {np.mean(list(scn_ref.values())):>8.4f} "
          f"{np.mean(list(rul_ref.values())):>8.4f} {overall:>8.4f}")

    with open(RUNS_ROOT / "cross_domain_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
