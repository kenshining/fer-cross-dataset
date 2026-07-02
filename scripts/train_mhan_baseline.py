"""
MHAN 基线训练 + 跨数据集评估

MHAN (Multi-Head Hybrid Attention Network) — Pattern Recognition 2026
Model: MixedFeatureNet backbone + ELA + Multi-head SEDDA attention
Input: 112×112 (half of standard 224)
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
from PIL import Image
from torchvision import transforms

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path("e:/scientific/小波/MHAN-code/MHAN-main")))

from networks.backbone import MHAN

DATA_ROOT = Path("e:/scientific/小波/data")
PROJECT_ROOT = Path("e:/scientific/小波")
RUNS_ROOT = _REPO / "runs" / "mhan_baseline"
BATCH_SIZE = 16
FACE_SIZE = 112  # MHAN uses 112x112
NUM_CLASSES = 7
EPOCHS = 60
LR = 5e-4
SEEDS = [42, 123]
NUM_HEAD = 2

os.makedirs(RUNS_ROOT, exist_ok=True)

# 标准化参数 (ImageNet)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# MHAN 预训练权重路径
PRETRAINED = PROJECT_ROOT / "MHAN-code" / "MHAN-main" / "pretrained" / "MFN_msceleb.pth"


def build_mhan_loader(dataset_name: str, split: str, batch_size: int, shuffle: bool):
    """构建 MHAN 用的 DataLoader (PIL 输出, 带 MHAN 预处理)。"""
    from src.dataset_registry import REGISTRY
    from src.dataset_fer2013 import FER2013Dataset, fer2013_collate_fn

    if dataset_name == "rafdb":
        if split == "train":
            ds = REGISTRY["rafdb"]["dataset_cls"](DATA_ROOT / "RAF-DB", split="train")
        else:
            ds = REGISTRY["rafdb"]["dataset_cls"](DATA_ROOT / "RAF-DB", split="test")
        collate = REGISTRY["rafdb"]["collate_fn"]
    elif dataset_name == "fer2013":
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

    # MHAN 用 ImageFolder 风格的 transform
    transform_fn = transforms.Compose([
        transforms.Resize((FACE_SIZE, FACE_SIZE)),
        transforms.RandomHorizontalFlip() if shuffle else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]) if shuffle else transforms.Compose([
        transforms.Resize((FACE_SIZE, FACE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    class MHANWrapper(torch.utils.data.Dataset):
        def __init__(self, ds, transform_fn):
            self.ds = ds
            self.t = transform_fn

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            pil, label = self.ds[idx]
            if not isinstance(pil, Image.Image):
                pil = pil.convert("RGB") if hasattr(pil, "convert") else Image.fromarray(np.array(pil))
            return self.t(pil), label

    wrapped = MHANWrapper(ds, transform_fn)
    return DataLoader(wrapped, batch_size=batch_size, shuffle=shuffle, num_workers=0)


class SmoothCrossEntropy(nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, logits, labels):
        num_classes = logits.shape[-1]
        alpha_div_k = self.alpha / num_classes
        target_probs = F.one_hot(labels, num_classes=num_classes).float() * (1. - self.alpha) + alpha_div_k
        return -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


class AttentionLoss(nn.Module):
    def forward(self, heads):
        if len(heads) < 2:
            return torch.tensor(0.0, device=heads[0].device, requires_grad=True)
        loss = 0.0
        cnt = 0
        for i in range(len(heads) - 1):
            for j in range(i + 1, len(heads)):
                loss += F.mse_loss(heads[i], heads[j])
                cnt += 1
        return loss / cnt


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0; total = 0
    tp = torch.zeros(NUM_CLASSES, device=device)
    fp = torch.zeros(NUM_CLASSES, device=device)
    fn = torch.zeros(NUM_CLASSES, device=device)

    for rgb, labels in loader:
        rgb, labels = rgb.to(device), labels.to(device)
        out, _, _ = model(rgb)
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}\nMHAN 基线: RAF-DB → 跨数据集 (2 seeds)")

    train_loader = build_mhan_loader("rafdb", "train", BATCH_SIZE, shuffle=True)
    val_loader = build_mhan_loader("rafdb", "test", BATCH_SIZE, shuffle=False)
    print(f"RAF-DB: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}")

    target_names = ["fer2013", "affectnet", "ckplus", "jaffe"]
    target_loaders = {}
    for tgt in target_names:
        try:
            split = "val" if tgt == "affectnet" else "test"
            target_loaders[tgt] = build_mhan_loader(tgt, split, BATCH_SIZE, shuffle=False)
            print(f"  {tgt}: {len(target_loaders[tgt].dataset)} 样本")
        except Exception as e:
            print(f"  [WARN] {tgt}: {e}")

    all_results = []

    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        print(f"\n{'='*50}\nSeed {seed}\n{'='*50}")

        model = MHAN(num_class=NUM_CLASSES, num_head=NUM_HEAD, pretrained=False).to(device)

        # 加载预训练 backbone (文件本身是 MixedFeatureNet 对象)
        if PRETRAINED.exists():
            pretrained_net = torch.load(PRETRAINED, map_location=device, weights_only=False)
            pretrained_features = nn.Sequential(*list(pretrained_net.children())[:-4])
            model.features.load_state_dict(pretrained_features.state_dict(), strict=True)
            print(f"  预训练权重已加载: {PRETRAINED}")

        criterion_cls = SmoothCrossEntropy(alpha=0.1)
        criterion_at = AttentionLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        best_val_f1 = 0.0
        best_state = None

        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss = 0.0; correct = 0; total_s = 0; iter_cnt = 0

            for rgb, labels in train_loader:
                rgb, labels = rgb.to(device), labels.to(device)
                optimizer.zero_grad()
                out, feat, heads = model(rgb)
                loss = criterion_cls(out, labels) + 0.1 * criterion_at(heads)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * labels.size(0)
                pred = out.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total_s += labels.size(0)
                iter_cnt += 1

            scheduler.step()
            val_m = evaluate(model, val_loader, device)

            if val_m["macro_f1"] > best_val_f1:
                best_val_f1 = val_m["macro_f1"]
                best_state = deepcopy(model.state_dict())

            if epoch % 15 == 0 or epoch == 1:
                print(f"  Epoch {epoch:2d}: loss={total_loss/total_s:.4f}, "
                      f"val_acc={val_m['acc']:.4f}, val_f1={val_m['macro_f1']:.4f}", flush=True)

        model.load_state_dict(best_state)
        torch.save({"model": best_state, "best_val_f1": best_val_f1},
                   RUNS_ROOT / f"mhan_rafdb_seed{seed}.pt")
        model.eval()
        print(f"  完成: best_val_f1={best_val_f1:.4f}")

        for tgt_name, tgt_loader in target_loaders.items():
            metrics = evaluate(model, tgt_loader, device)
            all_results.append({"seed": seed, "source": "rafdb", "target": tgt_name,
                                "acc": metrics["acc"], "macro_f1": metrics["macro_f1"]})
            print(f"    → {tgt_name}: f1={metrics['macro_f1']:.4f}")
        torch.cuda.empty_cache()

    # 汇总对比
    baseline_f1s = {"fer2013": 0.2969, "affectnet": 0.2491, "ckplus": 0.1739, "jaffe": 0.1534}
    scn_ref = {"fer2013": 0.3666, "affectnet": 0.3259, "ckplus": 0.2250, "jaffe": 0.1842}
    rul_ref = {"fer2013": 0.3677, "affectnet": 0.3250, "ckplus": 0.2271, "jaffe": 0.1384}

    print(f"\n{'='*60}")
    print(f"{'Target':<12} {'ResNet':>8} {'SCN':>8} {'RUL':>8} {'MHAN':>8}")
    print("-" * 50)
    for tgt in target_names:
        f1s = [r["macro_f1"] for r in all_results if r["target"] == tgt]
        if f1s and not np.isnan(f1s[0]):
            m = np.mean(f1s)
            print(f"{tgt:<12} {baseline_f1s[tgt]:>8.4f} {scn_ref[tgt]:>8.4f} "
                  f"{rul_ref[tgt]:>8.4f} {m:>8.4f}")

    overall = np.nanmean([r["macro_f1"] for r in all_results])
    print(f"\n{'平均':<12} {0.2183:>8.4f} {np.mean(list(scn_ref.values())):>8.4f} "
          f"{np.mean(list(rul_ref.values())):>8.4f} {overall:>8.4f}")

    with open(RUNS_ROOT / "cross_domain_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
