"""
RandAugment + MixUp 全量跨数据集评估
RAF-DB → AffectNet, CK+, JAFFE (每个 1 seed, 40 epochs)
"""
import sys, json, numpy as np
from copy import deepcopy
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms.functional as TF

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
        b=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.e=nn.Sequential(*list(b.children())[:-1])
        self.c=nn.Sequential(nn.Flatten(),nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.5),nn.Linear(256,NC))
    def forward(self,x): return self.c(self.e(x))

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

def randaug(rgb):
    B=rgb.size(0); out=[]
    for i in range(B):
        img=(rgb[i]*255).byte()
        if np.random.random()>.5: img=TF.adjust_brightness(img, float(.8+np.random.random()*.4))
        if np.random.random()>.5: img=TF.adjust_contrast(img, float(.8+np.random.random()*.4))
        if np.random.random()>.5: img=TF.adjust_sharpness(img, float(1.+np.random.random()))
        out.append(img.float()/255.)
    return torch.stack(out)

def mixup(rgb, alpha=0.2):
    B=rgb.size(0); lam=np.random.beta(alpha,alpha) if alpha>0 else 1.
    idx=torch.randperm(B,device=rgb.device); return lam*rgb+(1-lam)*rgb[idx]

def mixup_crit(crit, pred, y_a, y_b, lam):
    return lam*crit(pred,y_a)+(1-lam)*crit(pred,y_b)

def train_one(aug_fn=None, mixup_mode=False):
    torch.manual_seed(42); np.random.seed(42)
    model=R().to(DEVICE); opt=torch.optim.Adam(model.parameters(),lr=LR)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=0.5,patience=5)
    crit=nn.CrossEntropyLoss(); best_f1,best_state=0.,None
    raf=REGISTRY["rafdb"]; train_ds=raf["dataset_cls"](DATA_ROOT/"RAF-DB",split="train")
    tl=DataLoader(train_ds,batch_size=B,shuffle=True,num_workers=0,collate_fn=raf["collate_fn"])
    for ep in range(1,E+1):
        model.train()
        for bd,lb in tl:
            lb=lb.to(DEVICE)
            if isinstance(bd,list): rgb=torch.stack([pil_to_tensor01(center_crop_resize(p.convert("RGB"),FS)) for p in bd]).to(DEVICE)
            else: rgb=bd.to(DEVICE)
            if mixup_mode:
                lam=np.random.beta(.2,.2) if .2>0 else 1.; idx=torch.randperm(rgb.size(0),device=DEVICE)
                mixed=lam*rgb+(1-lam)*rgb[idx]; opt.zero_grad()
                loss=mixup_crit(crit,model(mixed),lb,lb[idx],lam); loss.backward(); opt.step()
            else:
                if aug_fn is not None: rgb=aug_fn(rgb)
                opt.zero_grad(); loss=crit(model(rgb),lb); loss.backward(); opt.step()
        m=ev(model,val_loader); sch.step(m["macro_f1"])
        if m["macro_f1"]>best_f1: best_f1,best_state=m["macro_f1"],deepcopy(model.state_dict())
    model.load_state_dict(best_state); model.eval()
    return model,best_f1

# Build RAF-DB val loader
raf=REGISTRY["rafdb"]; val_ds=raf["dataset_cls"](DATA_ROOT/"RAF-DB",split="test")
class VW(torch.utils.data.Dataset):
    def __init__(s,ds): s.ds=ds
    def __len__(s): return len(s.ds)
    def __getitem__(s,i): p,l=s.ds[i]; return pil_to_tensor01(center_crop_resize(p.convert("RGB"),FS)),l
val_loader=DataLoader(VW(val_ds),batch_size=B,shuffle=False)

# Build target loaders
def build_target(name):
    if name=="affectnet": t=REGISTRY["affectnet"]["dataset_cls"](DATA_ROOT/"AffectNet",split="val")
    elif name=="ckplus":  t=REGISTRY["ckplus"]["dataset_cls"](DATA_ROOT/"CK+")
    elif name=="jaffe":   t=REGISTRY["jaffe"]["dataset_cls"](DATA_ROOT/"Jaffe")
    else: raise ValueError(name)
    class W(torch.utils.data.Dataset):
        def __init__(s,ds): s.ds=ds
        def __len__(s): return len(s.ds)
        def __getitem__(s,i): p,l=s.ds[i]; return pil_to_tensor01(center_crop_resize(p.convert("RGB"),FS)),l
    return DataLoader(W(t),batch_size=B,shuffle=False)

targets={}
for t in ["affectnet","ckplus","jaffe"]:
    try: targets[t]=build_target(t); print(f"{t}: {len(targets[t].dataset)} samples")
    except Exception as e: print(f"{t}: FAILED - {e}")

# Load existing results
existing = OUT / "aug_results.json"
if existing.exists():
    with open(existing) as f: data = json.load(f)
else: data = {}

# RandAugment on other targets
print("\n=== RandAugment ===")
if "RandAugment" not in data: data["RandAugment"] = {}
m_r, _ = train_one(aug_fn=randaug)
for t, loader in targets.items():
    mc = ev(m_r, loader)
    data["RandAugment"][t] = {"cross_f1": mc["macro_f1"]}
    print(f"  → {t}: f1={mc['macro_f1']:.4f}")

# MixUp on other targets
print("\n=== MixUp ===")
if "MixUp" not in data: data["MixUp"] = {}
m_m, _ = train_one(mixup_mode=True)
for t, loader in targets.items():
    mc = ev(m_m, loader)
    data["MixUp"][t] = {"cross_f1": mc["macro_f1"]}
    print(f"  → {t}: f1={mc['macro_f1']:.4f}")

with open(existing, "w") as f: json.dump(data, f, indent=2)
print(f"\nSaved: {existing}")
print("All done!")
