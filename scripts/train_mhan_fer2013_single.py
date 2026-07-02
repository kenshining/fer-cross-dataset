"""
MHAN 单独训练: FER2013 source, seed 456, 40 epochs
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
RUNS_ROOT = _REPO / "runs" / "cross_source"
BATCH_SIZE = 16
NUM_CLASSES = 7
EPOCHS = 40
LR = 5e-4
SEED = 456
MEAN = [0.485, 0.456, 0.406]; STD = [0.229, 0.224, 0.225]
MHAN_PRETRAINED = Path("e:/scientific/小波/MHAN-code/MHAN-main/pretrained/MFN_msceleb.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(RUNS_ROOT, exist_ok=True)


class SmoothCrossEntropy(nn.Module):
    def __init__(self, alpha=0.1): super().__init__(); self.alpha = alpha
    def forward(self, logits, labels):
        nc = logits.shape[-1]; a = self.alpha / nc
        tp = F.one_hot(labels, nc).float() * (1 - self.alpha) + a
        return -(tp * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


class AttentionLoss(nn.Module):
    def forward(self, heads):
        if len(heads) < 2: return torch.tensor(0.0, device=heads[0].device, requires_grad=True)
        loss, cnt = 0.0, 0
        for i in range(len(heads)-1):
            for j in range(i+1, len(heads)):
                loss += F.mse_loss(heads[i], heads[j]); cnt += 1
        return loss / cnt


def build_fer_train(batch_size):
    import csv
    p = DATA_ROOT / "Fer2013" / "fer2013" / "fer2013.csv"
    class DS(torch.utils.data.Dataset):
        def __init__(s):
            s.rows = []
            with open(p) as f:
                for row in csv.DictReader(f):
                    if row.get("Usage","") == "Training":
                        s.rows.append((row["pixels"], int(row["emotion"])))
        def __len__(s): return len(s.rows)
        def __getitem__(s, i):
            px_str, l = s.rows[i]
            pix = np.fromstring(px_str, sep=" ", dtype=np.uint8)
            img = pix.reshape(48,48); img = np.stack([img]*3, axis=-1)
            pil = Image.fromarray(img).resize((112,112), Image.BILINEAR)
            t = transforms.ToTensor()(pil)
            t = transforms.Normalize(mean=MEAN, std=STD)(t)
            return t, l
    return DataLoader(DS(), batch_size=batch_size, shuffle=True, num_workers=0)


def build_fer_val(batch_size):
    import csv
    p = DATA_ROOT / "Fer2013" / "fer2013" / "fer2013.csv"
    class DS(torch.utils.data.Dataset):
        def __init__(s):
            s.rows = []
            with open(p) as f:
                for row in csv.DictReader(f):
                    if row.get("Usage","") == "PublicTest":
                        s.rows.append((row["pixels"], int(row["emotion"])))
        def __len__(s): return len(s.rows)
        def __getitem__(s, i):
            px_str, l = s.rows[i]
            pix = np.fromstring(px_str, sep=" ", dtype=np.uint8)
            img = pix.reshape(48,48); img = np.stack([img]*3, axis=-1)
            pil = Image.fromarray(img).resize((112,112), Image.BILINEAR)
            t = transforms.ToTensor()(pil)
            t = transforms.Normalize(mean=MEAN, std=STD)(t)
            return t, l
    return DataLoader(DS(), batch_size=batch_size, shuffle=False, num_workers=0)


def build_target_loader(name, batch_size):
    from src.dataset_registry import REGISTRY
    t = transforms.Compose([transforms.Resize((112,112)), transforms.ToTensor(),
                            transforms.Normalize(mean=MEAN, std=STD)])
    if name == "rafdb":
        ds = REGISTRY["rafdb"]["dataset_cls"](DATA_ROOT / "RAF-DB", split="test")
    elif name == "affectnet":
        ds = REGISTRY["affectnet"]["dataset_cls"](DATA_ROOT / "AffectNet", split="val")
    elif name == "ckplus":
        ds = REGISTRY["ckplus"]["dataset_cls"](DATA_ROOT / "CK+")
    elif name == "jaffe":
        ds = REGISTRY["jaffe"]["dataset_cls"](DATA_ROOT / "Jaffe")
    else: raise ValueError(name)

    class W(torch.utils.data.Dataset):
        def __init__(s): s.ds = ds
        def __len__(s): return len(s.ds)
        def __getitem__(s, i):
            p, l = s.ds[i]; p = p.convert("RGB") if hasattr(p,"convert") else p
            return t(p), l
    return DataLoader(W(), batch_size=batch_size, shuffle=False, num_workers=0)


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); correct = 0; total = 0
    tp = torch.zeros(NUM_CLASSES, device=DEVICE)
    fp = torch.zeros(NUM_CLASSES, device=DEVICE)
    fn = torch.zeros(NUM_CLASSES, device=DEVICE)
    for rgb, labels in loader:
        rgb, labels = rgb.to(DEVICE), labels.to(DEVICE)
        out, _, _ = model(rgb); pred = out.argmax(dim=1)
        correct += (pred == labels).sum().item(); total += labels.size(0)
        for c in range(NUM_CLASSES):
            tp[c] += ((pred==c)&(labels==c)).sum()
            fp[c] += ((pred==c)&(labels!=c)).sum()
            fn[c] += ((pred!=c)&(labels==c)).sum()
    acc = correct / max(total,1)
    prec = tp/(tp+fp+1e-8); rec = tp/(tp+fn+1e-8)
    f1 = 2*prec*rec/(prec+rec+1e-8)
    return {"acc": acc, "macro_f1": f1.mean().item()}


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    print(f"MHAN FER2013 source, seed={SEED}")

    train_loader = build_fer_train(BATCH_SIZE)
    val_loader = build_fer_val(BATCH_SIZE)
    print(f"FER2013: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}")

    targets = {"rafdb": None, "affectnet": None, "ckplus": None, "jaffe": None}
    for tgt in targets:
        try: targets[tgt] = build_target_loader(tgt, BATCH_SIZE)
        except: pass

    model = MHAN(num_class=NUM_CLASSES, num_head=2, pretrained=False).to(DEVICE)
    if MHAN_PRETRAINED.exists():
        pnet = torch.load(MHAN_PRETRAINED, map_location=DEVICE, weights_only=False)
        pf = nn.Sequential(*list(pnet.children())[:-4])
        model.features.load_state_dict(pf.state_dict(), strict=True)
        print("Pretrained loaded")

    crit_cls = SmoothCrossEntropy(0.1); crit_at = AttentionLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)
    best_f1, best_state = 0.0, None

    for ep in range(1, EPOCHS+1):
        model.train()
        for rgb, labels in train_loader:
            rgb, labels = rgb.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            out, feat, heads = model(rgb)
            loss = crit_cls(out, labels) + 0.1*crit_at(heads)
            loss.backward(); opt.step()
        sch.step()
        m = evaluate(model, val_loader)
        if m["macro_f1"] > best_f1:
            best_f1, best_state = m["macro_f1"], deepcopy(model.state_dict())
        if ep % 10 == 1: print(f"  ep{ep}: val_f1={m['macro_f1']:.4f}", flush=True)

    model.load_state_dict(best_state); model.eval()
    print(f"Done: best_f1={best_f1:.4f}")

    results = []
    for tgt, loader in targets.items():
        if loader:
            m = evaluate(model, loader)
            results.append({"seed": SEED, "source": "fer2013", "target": tgt,
                           "macro_f1": m["macro_f1"], "acc": m["acc"]})
            print(f"  → {tgt}: f1={m['macro_f1']:.4f}")

    # Append to existing results
    existing = RUNS_ROOT / "fer2013_source_results.json"
    if existing.exists():
        with open(existing) as f: old = json.load(f)
        old = [r for r in old if r.get("method") != "MHAN" or r.get("seed") != SEED]
    else:
        old = []
    for r in results: r["method"] = "MHAN"
    all_r = old + results
    with open(existing, "w") as f: json.dump(all_r, f, indent=2)
    print(f"Saved {len(all_r)} results to {existing}")


if __name__ == "__main__": main()
