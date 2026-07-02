"""
跨 source 验证: FER2013 → RAF-DB, AffectNet, CK+, JAFFE

方法: MHAN → SCN → ResNet-18 (依次训练)
每方法 1 seed, 40 epochs

用法:
  python fer_wavelet/scripts/train_cross_source.py

人脸预训练消融实验:
  python fer_wavelet/scripts/train_cross_source.py --ablation face_pretrain
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
from torchvision.models import resnet18, ResNet18_Weights

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path("e:/scientific/小波/MHAN-code/MHAN-main")))

DATA_ROOT = Path("e:/scientific/小波/data")
RUNS_ROOT = _REPO / "runs" / "cross_source"
BATCH_SIZE = 16
NUM_CLASSES = 7
EPOCHS = 40
LR = 1e-3
SEED = 123
SOURCE = "fer2013"  # 当前源数据集
TARGETS = ["rafdb", "affectnet", "ckplus", "jaffe"]

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
MHAN_PRETRAINED = Path("e:/scientific/小波/MHAN-code/MHAN-main/pretrained/MFN_msceleb.pth")

# 人脸预训练权重路径（MS1MV3 ArcFace ResNet-18）
FACE_WEIGHTS_PATH = _REPO / "pretrained" / "ms1mv3_arcface_r18.pth"

os.makedirs(RUNS_ROOT, exist_ok=True)


# ====================================================================
# Data
# ====================================================================

def build_fer2013_train_loader(batch_size: int, input_size: int = 224):
    """FER2013 训练集 DataLoader (lazy-loading tensor)."""
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
    from src.dataset_registry import REGISTRY
    from src.dataset_fer2013 import FER2013Dataset, fer2013_collate_fn

    if dataset_name == "rafdb":
        ds = REGISTRY["rafdb"]["dataset_cls"](DATA_ROOT / "RAF-DB", split="test")
        collate = REGISTRY["rafdb"]["collate_fn"]
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
        raise ValueError(f"Unknown: {dataset_name}")

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


# ====================================================================
# Models
# ====================================================================

class ResNetFER(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(512, 256),
                                        nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, NUM_CLASSES))
    def forward(self, x):
        return self.classifier(self.encoder(x))

class SCNFER(nn.Module):
    def __init__(self):
        super().__init__()
        from src.scn_model import SelfAttentionWeighting
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.alpha_module = SelfAttentionWeighting(512)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(512, 256),
                                        nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, NUM_CLASSES))
    def forward(self, x):
        f = self.encoder(x).view(x.size(0), -1)
        return self.classifier(f), f


class ResNetFER_Face(nn.Module):
    """使用 MS1MV3 ArcFace 预训练 iresnet18 backbone 的 ResNetFER。

    与 ResNetFER 使用相同的分类头和训练配置，仅 backbone 预训练域不同。
    用于消融实验：分离预训练域（人脸 vs ImageNet）对跨数据集泛化的影响。
    """
    def __init__(self, face_weights_path: str):
        super().__init__()
        from src.models import build_iresnet18_face_backbone
        self.encoder, _ = build_iresnet18_face_backbone(face_weights_path)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(512, 256),
                                        nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, NUM_CLASSES))
    def forward(self, x):
        return self.classifier(self.encoder(x))

# ====================================================================
# Evaluate
# ====================================================================

@torch.no_grad()
def evaluate(model, loader, device, mhan=False):
    model.eval()
    correct = 0; total = 0
    tp = torch.zeros(NUM_CLASSES, device=device)
    fp = torch.zeros(NUM_CLASSES, device=device)
    fn = torch.zeros(NUM_CLASSES, device=device)
    for batch_data, labels in loader:
        rgb, labels = batch_data.to(device), labels.to(device)
        if mhan:
            out, _, _ = model(rgb)
        else:
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


def train_resnet(model, train_loader, val_loader, device):
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


def train_scn(model, train_loader, val_loader, device):
    from src.scn_model import SCNLoss
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
    crit = SCNLoss(margin_1=0.07, margin_2=0.2, relabel_epoch=10)
    best_f1, best_state = 0.0, None
    for ep in range(1, EPOCHS + 1):
        model.train()
        for batch_data, labels in train_loader:
            rgb, labels = batch_data.to(device), labels.to(device)
            opt.zero_grad()
            logits, feat = model(rgb)
            loss, _, _, _ = crit(logits, feat, labels, model.alpha_module, ep)
            loss.backward(); opt.step()
        m = evaluate(model, val_loader, device)
        sch.step(m["macro_f1"])
        if m["macro_f1"] > best_f1:
            best_f1, best_state = m["macro_f1"], deepcopy(model.state_dict())
        if ep % 10 == 1:
            print(f"    SCN ep{ep}: val_f1={m['macro_f1']:.4f}")
    model.load_state_dict(best_state)
    return model, best_f1


def train_mhan(model, train_loader, val_loader, device):
    from train_mhan_baseline import SmoothCrossEntropy, AttentionLoss
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)
    crit_cls = SmoothCrossEntropy(0.1)
    crit_at = AttentionLoss()
    best_f1, best_state = 0.0, None
    for ep in range(1, EPOCHS + 1):
        model.train()
        for rgb, labels in train_loader:
            rgb, labels = rgb.to(device), labels.to(device)
            opt.zero_grad()
            out, feat, heads = model(rgb)
            loss = crit_cls(out, labels) + 0.1 * crit_at(heads)
            loss.backward(); opt.step()
        sch.step()
        m = evaluate(model, val_loader, device, mhan=True)
        if m["macro_f1"] > best_f1:
            best_f1, best_state = m["macro_f1"], deepcopy(model.state_dict())
        if ep % 10 == 1:
            print(f"    MHAN ep{ep}: val_f1={m['macro_f1']:.4f}")
    model.load_state_dict(best_state)
    return model, best_f1


# ====================================================================
# Main
# ====================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}\n跨 Source 验证: {SOURCE.upper()} → others")
    print(f"Targets: {TARGETS}")

    # FER2013 训练集
    train_loader_224 = build_fer2013_train_loader(BATCH_SIZE, 224)
    train_loader_112 = build_fer2013_train_loader(BATCH_SIZE, 112)
    print(f"  FER2013 train: {len(train_loader_224.dataset)} 样本")

    # 构建 FER2013 域内验证 (PublicTest)
    val_loader_224 = build_target_loader("fer2013", BATCH_SIZE, 224)
    val_loader_112 = build_target_loader("fer2013", BATCH_SIZE, 112)

    # 跨域 target loaders
    target_loaders_224 = {}
    target_loaders_112 = {}
    for tgt in TARGETS:
        try:
            target_loaders_224[tgt] = build_target_loader(tgt, BATCH_SIZE, 224)
            target_loaders_112[tgt] = build_target_loader(tgt, BATCH_SIZE, 112)
            print(f"  {tgt}: {len(target_loaders_224[tgt].dataset)} 样本")
        except Exception as e:
            print(f"  [WARN] {tgt}: {e}")

    all_results = []

    # ---- ResNet-18 ----
    print("\n=== ResNet-18 (FER2013 source) ===")
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = ResNetFER()
    model, best_f1 = train_resnet(model, train_loader_224, val_loader_224, device)
    print(f"  域内 best_f1={best_f1:.4f}")
    model.eval()
    for tgt in TARGETS:
        if tgt in target_loaders_224:
            m = evaluate(model, target_loaders_224[tgt], device)
            all_results.append({"method": "ResNet", "source": SOURCE, "target": tgt, "macro_f1": m["macro_f1"]})
            print(f"    → {tgt}: f1={m['macro_f1']:.4f}")
    torch.cuda.empty_cache()

    # ---- SCN ----
    print("\n=== SCN (FER2013 source) ===")
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = SCNFER()
    model, best_f1 = train_scn(model, train_loader_224, val_loader_224, device)
    print(f"  域内 best_f1={best_f1:.4f}")
    model.eval()
    for tgt in TARGETS:
        if tgt in target_loaders_224:
            m = evaluate(model, target_loaders_224[tgt], device)
            all_results.append({"method": "SCN", "source": SOURCE, "target": tgt, "macro_f1": m["macro_f1"]})
            print(f"    → {tgt}: f1={m['macro_f1']:.4f}")
    torch.cuda.empty_cache()

    # ---- MHAN ----
    print("\n=== MHAN (FER2013 source) ===")
    torch.manual_seed(SEED); np.random.seed(SEED)
    from networks.backbone import MHAN
    model = MHAN(num_class=NUM_CLASSES, num_head=2, pretrained=False).to(device)
    if MHAN_PRETRAINED.exists():
        pnet = torch.load(MHAN_PRETRAINED, map_location=device, weights_only=False)
        pfeat = nn.Sequential(*list(pnet.children())[:-4])
        model.features.load_state_dict(pfeat.state_dict(), strict=True)
    model, best_f1 = train_mhan(model, train_loader_112, val_loader_112, device)
    print(f"  域内 best_f1={best_f1:.4f}")
    model.eval()
    for tgt in TARGETS:
        if tgt in target_loaders_112:
            m = evaluate(model, target_loaders_112[tgt], device, mhan=True)
            all_results.append({"method": "MHAN", "source": SOURCE, "target": tgt, "macro_f1": m["macro_f1"]})
            print(f"    → {tgt}: f1={m['macro_f1']:.4f}")
    torch.cuda.empty_cache()

    # ---- 汇总 ----
    print(f"\n{'='*60}")
    print(f"FER2013 source → cross-dataset")
    print(f"{'Target':<12} {'ResNet':>8} {'SCN':>8} {'MHAN':>8}")
    print("-" * 45)
    for tgt in TARGETS:
        vals = {}
        for r in all_results:
            if r["target"] == tgt:
                vals[r["method"]] = r["macro_f1"]
        print(f"{tgt:<12} {vals.get('ResNet',0):>8.4f} {vals.get('SCN',0):>8.4f} {vals.get('MHAN',0):>8.4f}")

    with open(RUNS_ROOT / f"{SOURCE}_source_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n保存: {RUNS_ROOT / f'{SOURCE}_source_results.json'}")


if __name__ == "__main__":
    main()
