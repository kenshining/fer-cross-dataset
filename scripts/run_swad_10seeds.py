"""
SWAD (SWA Dense) 10-seed training (RAF-DB source, 4 targets).
Dense-to-sparse weight averaging for domain generalization.
All 10 seeds: 10, 42, 89, 123, 456, 781, 789, 999, 1337, 2026.
"""
import sys, json, numpy as np, time
from copy import deepcopy
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
from src.dataset_registry import REGISTRY
from src.preprocess import center_crop_resize, pil_to_tensor01

DEVICE = torch.device("cuda")
B, LR, EPOCHS, NC, FS = 16, 1e-3, 40, 7, 224
DATA_ROOT = Path("e:/scientific/小波/data")
OUT = _REPO / "runs" / "swad_10seeds"
OUT.mkdir(parents=True, exist_ok=True)

ALL_SEEDS = [10, 42, 89, 123, 456, 781, 789, 999, 1337, 2026]
TARGETS = ["fer2013","affectnet","ckplus","jaffe"]
MEAN = [0.485,0.456,0.406]; STD = [0.229,0.224,0.225]

# SWAD config
SWAD_START = 20      # start collecting checkpoints from epoch 20
SWAD_R = 0.5          # density: keep 1 out of every 2 checkpoints (sparser ensemble)
SWAD_N_CONVERGE = 3   # stop SWA when loss doesn't improve for n epochs

class RN(nn.Module):
    def __init__(self):
        super().__init__()
        b = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.e = nn.Sequential(*list(b.children())[:-1])
        self.c = nn.Sequential(nn.Flatten(), nn.Linear(512,256), nn.ReLU(),
                               nn.Dropout(0.5), nn.Linear(256,NC))
    def forward(self,x): return self.c(self.e(x))

class SWAD:
    """Dense-to-sparse SWA: collect all checkpoints, then keep every 1/r."""
    def __init__(self, start_epoch=20, r=0.5):
        self.checkpoints = []
        self.start = start_epoch
        self.r = r

    def update(self, model):
        self.checkpoints.append(deepcopy(model.state_dict()))

    def get_swa_model(self, device):
        n = len(self.checkpoints)
        # Dense-to-sparse: keep every ceil(1/r) checkpoints
        keep_step = max(1, int(1/self.r))
        keep_idx = list(range(0, n, keep_step))
        avg_state = deepcopy(self.checkpoints[keep_idx[0]])
        for i in keep_idx[1:]:
            for k in avg_state:
                avg_state[k] = avg_state[k].to(device) + self.checkpoints[i][k].to(device)
        for k in avg_state:
            avg_state[k] /= len(keep_idx)
        return avg_state

def build_rafdb_train(bs):
    ds = REGISTRY["rafdb"]["dataset_cls"](DATA_ROOT/"RAF-DB", split="train")
    return DataLoader(ds, batch_size=bs, shuffle=True,
                      collate_fn=REGISTRY["rafdb"]["collate_fn"], num_workers=0)

def build_target_loader(name, bs):
    if name == "fer2013":
        import tempfile, csv
        from src.dataset_fer2013 import FER2013Dataset, fer2013_collate_fn
        cp = DATA_ROOT/"Fer2013"/"fer2013"/"fer2013.csv"
        tmp = tempfile.NamedTemporaryFile(mode="w",suffix=".csv",delete=False,encoding="utf-8")
        tmp.write("emotion,pixels\n")
        with open(cp) as f:
            for line in f:
                if "PublicTest" in line:
                    p=line.strip().split(",",2)
                    if len(p)>=2: tmp.write(f"{p[0]},{p[1]}\n")
        tmp.close()
        ds = FER2013Dataset(Path(tmp.name))
        return DataLoader(ds,batch_size=bs,shuffle=False,collate_fn=fer2013_collate_fn,num_workers=0)
    elif name == "affectnet":
        ds=REGISTRY["affectnet"]["dataset_cls"](DATA_ROOT/"AffectNet",split="val")
        return DataLoader(ds,batch_size=bs,shuffle=False,collate_fn=REGISTRY["affectnet"]["collate_fn"],num_workers=0)
    elif name == "ckplus":
        ds=REGISTRY["ckplus"]["dataset_cls"](DATA_ROOT/"CK+")
        return DataLoader(ds,batch_size=bs,shuffle=False,collate_fn=REGISTRY["ckplus"]["collate_fn"],num_workers=0)
    elif name == "jaffe":
        ds=REGISTRY["jaffe"]["dataset_cls"](DATA_ROOT/"Jaffe")
        return DataLoader(ds,batch_size=bs,shuffle=False,collate_fn=REGISTRY["jaffe"]["collate_fn"],num_workers=0)
    raise ValueError(name)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    tp=torch.zeros(NC,device=DEVICE); fp=torch.zeros(NC,device=DEVICE)
    fn=torch.zeros(NC,device=DEVICE); tot=0
    for batch in loader:
        imgs, labels = batch
        if isinstance(imgs, list):
            rgb=torch.stack([pil_to_tensor01(center_crop_resize(p.convert("RGB"),FS)) for p in imgs]).to(DEVICE)
        else:
            rgb=imgs.to(DEVICE)
        labels=labels.to(DEVICE)
        pr=model(rgb).argmax(dim=1); tot+=labels.size(0)
        for c in range(NC):
            tp[c]+=((pr==c)&(labels==c)).sum(); fp[c]+=((pr==c)&(labels!=c)).sum()
            fn[c]+=((pr!=c)&(labels==c)).sum()
    pre=tp/(tp+fp+1e-8); rec=tp/(tp+fn+1e-8)
    return {"macro_f1":(2*pre*rec/(pre+rec+1e-8)).mean().item(),
            "acc":(tp.sum()/max(tot,1)).item()}

def train_swad(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    tl = build_rafdb_train(B)
    model = RN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    crit = nn.CrossEntropyLoss()
    swad = SWAD(start_epoch=SWAD_START, r=SWAD_R)
    best_val_f1 = 0; best_state = None

    for ep in range(EPOCHS):
        model.train(); total_loss=0; nb=0
        for batch in tl:
            imgs, labels = batch
            if isinstance(imgs, list):
                rgb=torch.stack([pil_to_tensor01(center_crop_resize(p.convert("RGB"),FS)) for p in imgs]).to(DEVICE)
            else:
                rgb=imgs.to(DEVICE)
            labels=labels.to(DEVICE)
            opt.zero_grad(); loss = crit(model(rgb), labels)
            loss.backward(); opt.step()
            total_loss += loss.item(); nb += 1
        avg = total_loss/max(nb,1); sched.step(avg)

        if ep >= SWAD_START:
            swad.update(model)

        if (ep+1)%10==0: print(f"  Epoch {ep+1}/{EPOCHS}, loss={avg:.4f}")

    # Use SWA ensemble for final model
    swa_state = swad.get_swa_model(DEVICE)
    model.load_state_dict(swa_state)

    results = {}
    for tgt in TARGETS:
        try:
            loader = build_target_loader(tgt, B)
            results[tgt] = round(evaluate(model, loader)["macro_f1"], 4)
        except Exception as e:
            results[tgt] = None; print(f"  [ERR] {tgt}: {e}")
    results["mean"] = round(float(np.mean([v for v in results.values() if v])), 4)
    return results

print("="*60)
print("SWAD 10-Seed Training")
print(f"Seeds: {ALL_SEEDS}")
print(f"Config: start_epoch={SWAD_START}, r={SWAD_R}")
print("="*60)

all_results = {}
res_path = OUT / "swad_results.json"
if res_path.exists():
    with open(res_path) as f: all_results = json.load(f)

for seed in ALL_SEEDS:
    key = f"SWAD_s{seed}"
    if key in all_results:
        print(f"  [{key}] SKIP")
        continue
    t0=time.time()
    print(f"  [{key}] Training...")
    try:
        r = train_swad(seed)
        all_results[key] = r
        print(f"  [{key}] Mean={r['mean']:.4f} ({time.time()-t0:.0f}s)")
        with open(res_path,"w") as f: json.dump(all_results, f, indent=2)
    except Exception as e:
        print(f"  [{key}] ERROR: {e}")

vals=[v["mean"] for v in all_results.values() if "mean" in v]
print(f"\nSWAD: {len(vals)} seeds, Mean={np.mean(vals):.4f} +/- {np.std(vals):.4f}")
