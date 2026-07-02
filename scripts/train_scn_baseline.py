"""
SCN 基线训练 + 跨数据集评估

用法:
  python fer_wavelet/scripts/train_scn_baseline.py

输出:
  runs/scn_baseline/scn_rafdb/best.pt
  runs/scn_baseline/cross_domain_results.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18, ResNet18_Weights

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.scn_model import SCNLoss, SelfAttentionWeighting
from src.preprocess import center_crop_resize, pil_to_tensor01
from src.dataset_registry import REGISTRY

# ---- 配置 ----
DATA_ROOT = Path("e:/scientific/小波/data")
RUNS_ROOT = _REPO / "runs" / "scn_baseline"
BATCH_SIZE = 16
FACE_SIZE = 224
NUM_CLASSES = 7
EPOCHS = 40
LR = 1e-3
MARGIN_1 = 0.07
MARGIN_2 = 0.2
RELABEL_EPOCH = 10
LAMBDA_RR = 0.1
SEEDS = [42, 123]  # 2 seeds 快速验证

os.makedirs(RUNS_ROOT, exist_ok=True)


# ====================================================================
# SCN 模型
# ====================================================================

class SCNModel(nn.Module):
    """SCN: ResNet-18 + SelfAttentionWeighting + Classifier."""

    def __init__(self, num_classes: int = 7):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        backbone = resnet18(weights=weights)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])  # (B,512,1,1)
        self.feat_dim = 512

        # 重要性加权模块
        self.alpha_module = SelfAttentionWeighting(self.feat_dim)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor):
        f = self.encoder(x)  # (B, 512, 1, 1)
        f_flat = f.view(f.size(0), -1)  # (B, 512)
        logits = self.classifier(f)
        return logits, f_flat


# ====================================================================
# 数据
# ====================================================================

def build_loader(dataset_name: str, split: str, batch_size: int, shuffle: bool):
    """构建指定数据集的 DataLoader。"""
    if dataset_name == "rafdb":
        ds = REGISTRY["rafdb"]["dataset_cls"](DATA_ROOT / "RAF-DB", split=split)
        collate = REGISTRY["rafdb"]["collate_fn"]
    elif dataset_name == "fer2013":
        from src.dataset_fer2013 import FER2013Dataset, fer2013_collate_fn
        # Use original fer2013.csv, filter PublicTest rows into a temp file
        import tempfile
        csv_path = DATA_ROOT / "Fer2013" / "fer2013" / "fer2013.csv"
        # Write filtered CSV with only PublicTest rows
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
        ds = REGISTRY["affectnet"]["dataset_cls"](DATA_ROOT / "AffectNet", split="val")
        collate = REGISTRY["affectnet"]["collate_fn"]
    elif dataset_name == "ckplus":
        ds = REGISTRY["ckplus"]["dataset_cls"](DATA_ROOT / "CK+")
        collate = REGISTRY["ckplus"]["collate_fn"]
    elif dataset_name == "jaffe":
        ds = REGISTRY["jaffe"]["dataset_cls"](DATA_ROOT / "Jaffe")
        collate = REGISTRY["jaffe"]["collate_fn"]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      collate_fn=collate)


# ====================================================================
# 评估
# ====================================================================

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
            rgb_list = [pil_to_tensor01(center_crop_resize(p.convert("RGB"), FACE_SIZE))
                        for p in batch_data]
            rgb = torch.stack(rgb_list).to(device)

        logits, _ = model(rgb)
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


# ====================================================================
# 训练
# ====================================================================

def train_scn(model, train_loader, val_loader, device, seed, run_dir: Path):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6,
    )
    criterion = SCNLoss(
        margin_1=MARGIN_1, margin_2=MARGIN_2,
        relabel_epoch=RELABEL_EPOCH, lambda_rr=LAMBDA_RR,
    )

    best_val_f1 = 0.0
    best_state = None
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_ce = 0.0; total_rr = 0.0; total_loss = 0.0
        correct = 0; total = 0
        alpha_vals = []

        for batch_data, labels in train_loader:
            labels = labels.to(device)
            if isinstance(batch_data, torch.Tensor):
                rgb = batch_data.to(device)
            else:
                rgb_list = [pil_to_tensor01(center_crop_resize(p.convert("RGB"), FACE_SIZE))
                            for p in batch_data]
                rgb = torch.stack(rgb_list).to(device)

            optimizer.zero_grad()
            logits, features = model(rgb)

            loss, ce, rr, alpha = criterion(
                logits, features, labels, model.alpha_module, epoch,
            )
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            total_ce += ce.item() * bs
            total_rr += rr.item() * bs
            total_loss += loss.item() * bs
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += bs
            alpha_vals.append(alpha.detach().mean().item())

        # 域内评估
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["macro_f1"])

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = deepcopy(model.state_dict())

        history.append({
            "epoch": epoch, "ce_loss": total_ce / total, "rr_loss": total_rr / total,
            "train_acc": correct / total, "alpha_mean": np.mean(alpha_vals),
            "val_acc": val_metrics["acc"], "val_f1": val_metrics["macro_f1"],
        })

        if epoch % 10 == 0 or epoch == 1:
            relabel_on = "ON" if epoch >= RELABEL_EPOCH else "OFF"
            print(f"  Epoch {epoch:2d}: ce={total_ce/total:.4f}, rr={total_rr/total:.4f}, "
                  f"α={np.mean(alpha_vals):.3f}, val_f1={val_metrics['macro_f1']:.4f}, "
                  f"relabel={relabel_on}", flush=True)

    # 保存最佳模型
    model.load_state_dict(best_state)
    torch.save({
        "model": best_state,
        "best_val_f1": best_val_f1,
        "history": history,
    }, run_dir / "best.pt")

    # 保存历史
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"  训练完成: best_val_f1={best_val_f1:.4f}", flush=True)
    return model


# ====================================================================
# 主函数
# ====================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print("=" * 70)
    print("SCN 基线: 训练 RAF-DB → 跨数据集评估")
    print(f"Batch={BATCH_SIZE}, Epochs={EPOCHS}, LR={LR}")
    print(f"SCN 参数: margin_1={MARGIN_1}, margin_2={MARGIN_2}, "
          f"relabel_epoch={RELABEL_EPOCH}")
    print("=" * 70)

    # 构建 RAF-DB 训练/验证 loader
    print("\n[1] 构建 RAF-DB DataLoader...")
    train_loader = build_loader("rafdb", "train", BATCH_SIZE, shuffle=True)
    val_loader = build_loader("rafdb", "test", BATCH_SIZE, shuffle=False)
    print(f"  训练: {len(train_loader.dataset)} 样本, 验证: {len(val_loader.dataset)} 样本")

    # 跨数据集目标 loader
    target_names = ["fer2013", "affectnet", "ckplus", "jaffe"]
    target_loaders = {}
    print("\n[2] 构建跨数据集测试 Loader...")
    for tgt in target_names:
        try:
            if tgt == "affectnet":
                loader = build_loader(tgt, "val", BATCH_SIZE, shuffle=False)
            else:
                loader = build_loader(tgt, "test", BATCH_SIZE, shuffle=False)
            target_loaders[tgt] = loader
            print(f"  {tgt}: {len(loader.dataset)} 样本")
        except Exception as e:
            print(f"  [WARN] {tgt}: {e}")

    # 逐 seed 训练
    all_cross_results = []

    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        run_dir = RUNS_ROOT / f"scn_rafdb_seed{seed}"
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n{'='*50}")
        print(f"Seed {seed}")
        print(f"{'='*50}")

        model = SCNModel(num_classes=NUM_CLASSES)
        model = train_scn(model, train_loader, val_loader, device, seed, run_dir)
        model.eval()

        # 跨数据集评估
        print(f"\n  跨数据集评估:")
        for tgt_name, tgt_loader in target_loaders.items():
            metrics = evaluate(model, tgt_loader, device)
            all_cross_results.append({
                "seed": seed, "source": "rafdb", "target": tgt_name,
                "acc": metrics["acc"], "macro_f1": metrics["macro_f1"],
            })
            print(f"    → {tgt_name}: acc={metrics['acc']:.4f}, "
                  f"macro_f1={metrics['macro_f1']:.4f}")

        torch.cuda.empty_cache()

    # 汇总
    print(f"\n{'=' * 70}")
    print("SCN 跨数据集评估汇总 (RAF-DB → X)")
    print("=" * 70)

    for tgt in target_names:
        f1s = [r["macro_f1"] for r in all_cross_results if r["target"] == tgt]
        if f1s:
            print(f"  {tgt}: macro_f1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}  "
                  f"[{', '.join(f'{x:.4f}' for x in f1s)}]")

    # 与 ResNet-18 基线对比 (来自 H₂ 数据)
    baseline_f1s = {
        "fer2013": 0.2969, "affectnet": 0.2491, "ckplus": 0.1739, "jaffe": 0.1534,
    }
    print(f"\n  vs. ResNet-18 基线:")
    scn_vals = []
    base_vals = []
    for tgt in target_names:
        vals = [r["macro_f1"] for r in all_cross_results if r["target"] == tgt]
        if not vals or all(np.isnan(v) for v in vals):
            continue
        scn_mean = np.nanmean(vals)
        baseline = baseline_f1s.get(tgt, 0)
        delta = scn_mean - baseline
        scn_vals.append(scn_mean)
        base_vals.append(baseline)
        print(f"    {tgt}: SCN={scn_mean:.4f}, Baseline={baseline:.4f}, Δ={delta:+.4f}")

    if scn_vals:
        overall_scn = np.mean(scn_vals)
        overall_baseline = np.mean(base_vals)
        overall_delta = overall_scn - overall_baseline
    else:
        overall_scn = overall_baseline = overall_delta = 0
    print(f"\n  总体: SCN={overall_scn:.4f}, Baseline={overall_baseline:.4f}, "
          f"Δ={overall_delta:+.4f}")

    # 判断
    if overall_delta > 0.02:
        print(f"\n  ✅ SCN 显著优于 Baseline (Δ={overall_delta:+.4f} > 0.02) → 方向可行!")
    elif overall_delta > 0.005:
        print(f"\n  ⚠️ SCN 略优于 Baseline (Δ={overall_delta:+.4f}) → 可继续但有风险")
    else:
        print(f"\n  ❌ SCN 未显著优于 Baseline (Δ={overall_delta:+.4f}) → 谨慎判断")

    # 保存结果
    results_path = RUNS_ROOT / "cross_domain_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_cross_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {results_path}")


if __name__ == "__main__":
    main()
