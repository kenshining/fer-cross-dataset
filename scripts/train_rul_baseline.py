"""
RUL 基线训练 + 跨数据集评估

用法:
  python fer_wavelet/scripts/train_rul_baseline.py
"""
from __future__ import annotations

import json, os, sys, time, tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.rul_model import RULModel
from src.preprocess import center_crop_resize, pil_to_tensor01
from src.dataset_registry import REGISTRY

DATA_ROOT = Path("e:/scientific/小波/data")
RUNS_ROOT = _REPO / "runs" / "rul_baseline"
BATCH_SIZE = 16
FACE_SIZE = 224
NUM_CLASSES = 7
EPOCHS = 40
LR = 1e-3
SEEDS = [42, 123]
LAMBDA_U = 0.1   # 不确定性损失权重
LAMBDA_AU = 0.5  # Add-up损失权重

os.makedirs(RUNS_ROOT, exist_ok=True)


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
        logits, _, _ = model(rgb)
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


def train_rul(model, train_loader, val_loader, device, seed, run_dir: Path):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6)
    ce_criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0; correct = 0; total_s = 0
        total_ce = 0.0; total_au = 0.0; total_u = 0.0

        for batch_data, labels in train_loader:
            labels = labels.to(device)
            if isinstance(batch_data, torch.Tensor):
                rgb = batch_data.to(device)
            else:
                rgb_list = [pil_to_tensor01(center_crop_resize(p.convert("RGB"), FACE_SIZE)) for p in batch_data]
                rgb = torch.stack(rgb_list).to(device)

            B = rgb.size(0)

            # 标准前向
            optimizer.zero_grad()
            logits, features, uncertainty = model(rgb)

            # 随机配对
            idx = torch.randperm(B, device=device)
            lam = torch.rand(B, 1, device=device)  # (B, 1)

            # 特征空间mixup
            features_mixed = lam * features + (1 - lam) * features[idx]

            # 混合特征通过分类头
            logits_mixed = model.classifier(features_mixed)

            # CE (原始)
            ce_loss = ce_criterion(logits, labels)

            # Add-up (混合): 两个标签加权
            log_probs_mix = F.log_softmax(logits_mixed, dim=1)
            au_a = -log_probs_mix[torch.arange(B, device=device), labels]
            au_b = -log_probs_mix[torch.arange(B, device=device), labels[idx]]
            au_loss = (lam.squeeze() * au_a + (1 - lam.squeeze()) * au_b).mean()

            # 不确定性正则
            u_loss_val = 0.0
            if LAMBDA_U > 0:
                ce_per = F.cross_entropy(logits, labels, reduction="none")
                u_loss_val = -(uncertainty * ce_per.detach()).mean()

            loss = ce_loss + LAMBDA_AU * au_loss + LAMBDA_U * u_loss_val
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            total_ce += ce_loss.item() * B
            total_au += au_loss.item() * B
            total_u += u_loss_val.item() * B if isinstance(u_loss_val, torch.Tensor) else u_loss_val
            pred = logits[:B].argmax(dim=1)
            correct += (pred == labels).sum().item()
            total_s += B

        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["macro_f1"])

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = deepcopy(model.state_dict())

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d}: ce={total_ce/total_s:.4f}, au={total_au/total_s:.4f}, "
                  f"val_f1={val_metrics['macro_f1']:.4f}", flush=True)

    model.load_state_dict(best_state)
    torch.save({"model": best_state, "best_val_f1": best_val_f1}, run_dir / "best.pt")
    print(f"  训练完成: best_val_f1={best_val_f1:.4f}", flush=True)
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}\nRUL 基线: RAF-DB → 跨数据集 (2 seeds)")

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
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        run_dir = RUNS_ROOT / f"rul_rafdb_seed{seed}"
        os.makedirs(run_dir, exist_ok=True)
        print(f"\nSeed {seed}")
        model = RULModel(num_classes=NUM_CLASSES)
        model = train_rul(model, train_loader, val_loader, device, seed, run_dir)
        model.eval()

        for tgt_name, tgt_loader in target_loaders.items():
            metrics = evaluate(model, tgt_loader, device)
            all_results.append({"seed": seed, "source": "rafdb", "target": tgt_name,
                                "acc": metrics["acc"], "macro_f1": metrics["macro_f1"]})
            print(f"    → {tgt_name}: f1={metrics['macro_f1']:.4f}")
        torch.cuda.empty_cache()

    # 汇总
    baseline_f1s = {"fer2013": 0.2969, "affectnet": 0.2491, "ckplus": 0.1739, "jaffe": 0.1534}
    print(f"\n{'='*60}\nRUL vs Baseline")
    scn_ref = {"fer2013": 0.3666, "affectnet": 0.3259, "ckplus": 0.2250, "jaffe": 0.1842}
    for tgt in target_names:
        f1s = [r["macro_f1"] for r in all_results if r["target"] == tgt]
        if f1s and not np.isnan(f1s[0]):
            m = np.mean(f1s)
            print(f"  {tgt}: RUL={m:.4f}, SCN={scn_ref[tgt]:.4f}, "
                  f"Baseline={baseline_f1s[tgt]:.4f}, Δ_RUL={m-baseline_f1s[tgt]:+.4f}")

    overall = np.mean([r["macro_f1"] for r in all_results if not np.isnan(r["macro_f1"])])
    print(f"\n  总体: RUL={overall:.4f}, SCN={np.mean(list(scn_ref.values())):.4f}, "
          f"Baseline=0.2183")

    with open(RUNS_ROOT / "cross_domain_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
