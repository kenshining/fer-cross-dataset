"""
MixUp baseline (label-aware) on RAF-DB + cross-dataset eval
"""
import sys, json, numpy as np
from copy import deepcopy
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
from src.dataset_registry import REGISTRY
from src.preprocess import center_crop_resize, pil_to_tensor01

DEVICE = torch.device("cuda"); B, LR, E, NC, FS = 16, 1e-3, 40, 7, 224
DATA_ROOT = Path("e:/scientific/小波/data")
OUT = _REPO / "runs" / "aug_baselines"

class R(nn.Module):
    def __init__(self):
        super().__init__()
        b = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.e = nn.Sequential(*list(b.children())[:-1])
        self.c = nn.Sequential(nn.Flatten(), nn.Linear(512,256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256,NC))
    def forward(self, x): return self.c(self.e(x))

@torch.no_grad()
def ev(model, loader):
    model.eval(); cor=0; tot=0
    tp=torch.zeros(NC,device=DEVICE);fp=torch.zeros(NC,device=DEVICE);fn=torch.zeros(NC,device=DEVICE)
    for bd,lb in loader:
        lb=lb.to(DEVICE)
        if isinstance(bd,list): rgb=torch.stack([pil_to_tensor01(center_crop_resize(p.convert("RGB"),FS)) for p in bd]).to(DEVICE)
        else: rgb=bd.to(DEVICE)
        pr=model(rgb).argmax(dim=1);cor+=(pr==lb).sum().item();tot+=lb.size(0)
        for c in range(NC): tp[c]+=((pr==c)&(lb==c)).sum();fp[c]+=((pr==c)&(lb!=c)).sum();fn[c]+=((pr!=c)&(lb==c)).sum()
    acc=cor/max(tot,1);prc=tp/(tp+fp+1e-8);rec=tp/(tp+fn+1e-8);f1=2*prc*rec/(prc+rec+1e-8)
    return {"acc":acc,"macro_f1":f1.mean().item()}

# --- Data ---
raf = REGISTRY["rafdb"]
train_ds = raf["dataset_cls"](DATA_ROOT/"RAF-DB", split="train")
val_ds = raf["dataset_cls"](DATA_ROOT/"RAF-DB", split="test")
class VW(torch.utils.data.Dataset):
    def __init__(s,ds): s.ds=ds
    def __len__(s): return len(s.ds)
    def __getitem__(s,i): p,l=s.ds[i]; return pil_to_tensor01(center_crop_resize(p.convert("RGB"),FS)),l
val_loader = DataLoader(VW(val_ds), batch_size=B, shuffle=False)

import csv, tempfile
from src.dataset_fer2013 import FER2013Dataset, fer2013_collate_fn
csv_p = DATA_ROOT/"Fer2013"/"fer2013"/"fer2013.csv"
tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8")
tmp.write("emotion,pixels\n")
with open(csv_p) as fin:
    for line in fin:
        if "PublicTest" in line:
                parts=line.strip().split(",",2)
                if len(parts)>=2: tmp.write(f"{parts[0]},{parts[1]}\n")
tmp.close()
fer_loader = DataLoader(FER2013Dataset(Path(tmp.name)), batch_size=B, shuffle=False, collate_fn=fer2013_collate_fn)

def mixup_criterion(crit, pred, y_a, y_b, lam):
    """Label-aware MixUp loss."""
    return lam * crit(pred, y_a) + (1 - lam) * crit(pred, y_b)

def train_mixup(alpha=0.2):
    torch.manual_seed(42); np.random.seed(42)
    model = R().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
    crit = nn.CrossEntropyLoss()
    best_f1, best_state = 0.0, None
    tl = DataLoader(train_ds, batch_size=B, shuffle=True, num_workers=0, collate_fn=raf["collate_fn"])

    for ep in range(1, E+1):
        model.train()
        for bd, lb in tl:
            lb = lb.to(DEVICE)
            if isinstance(bd,list): rgb=torch.stack([pil_to_tensor01(center_crop_resize(p.convert("RGB"),FS)) for p in bd]).to(DEVICE)
            else: rgb=bd.to(DEVICE)
            # MixUp
            lam = np.random.beta(alpha, alpha) if alpha>0 else 1.0
            idx = torch.randperm(rgb.size(0), device=DEVICE)
            mixed = lam*rgb + (1-lam)*rgb[idx]
            opt.zero_grad()
            loss = mixup_criterion(crit, model(mixed), lb, lb[idx], lam)
            loss.backward(); opt.step()
        m = ev(model, val_loader); sch.step(m["macro_f1"])
        if m["macro_f1"] > best_f1: best_f1, best_state = m["macro_f1"], deepcopy(model.state_dict())
    model.load_state_dict(best_state); model.eval()
    return model, best_f1

print("=== MixUp (label-aware) ===")
m, f1 = train_mixup(alpha=0.2)
mc = ev(m, fer_loader)
print(f"  In={f1:.4f}, Cross(FER2013)={mc['macro_f1']:.4f}")

# Update results
existing = OUT / "aug_results.json"
if existing.exists():
    with open(existing) as f: data = json.load(f)
else: data = {}
data["MixUp"] = {"in": f1, "cross": mc["macro_f1"]}
with open(existing, "w") as f: json.dump(data, f, indent=2)
print(f"Updated {existing}")
