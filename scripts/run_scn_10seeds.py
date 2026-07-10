"""
SCN 10-seed training (RAF-DB source, 4 targets).
New seeds: 10, 89, 781, 999, 1337, 2026 (existing: 42, 123, 456, 789).
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
from src.scn_model import SelfAttentionWeighting

DEVICE = torch.device("cuda")
B, LR, EPOCHS, NC, FS = 16, 1e-3, 40, 7, 224
DATA_ROOT = Path("e:/scientific/小波/data")
OUT = _REPO / "runs" / "scn_10seeds"
OUT.mkdir(parents=True, exist_ok=True)

NEW_SEEDS = [10, 89, 781, 999, 1337, 2026]
TARGETS = ["fer2013","affectnet","ckplus","jaffe"]
MEAN = [0.485,0.456,0.406]; STD = [0.229,0.224,0.225]

class SCN(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.alpha_module = SelfAttentionWeighting(512)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(512,256), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(256,NC))
        self._relabel_buffer = {}

    def forward(self, x, epoch=None):
        f = self.encoder(x).view(x.size(0),-1)
        a = self.alpha_module(f)
        logits = self.classifier(f)
        if self.training and epoch is not None and epoch >= 10:
            return logits, a
        return logits, a

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
        logits, _ = model(rgb)
        pr=logits.argmax(dim=1); tot+=labels.size(0)
        for c in range(NC):
            tp[c]+=((pr==c)&(labels==c)).sum(); fp[c]+=((pr==c)&(labels!=c)).sum()
            fn[c]+=((pr!=c)&(labels==c)).sum()
    pre=tp/(tp+fp+1e-8); rec=tp/(tp+fn+1e-8)
    return {"macro_f1":(2*pre*rec/(pre+rec+1e-8)).mean().item(),
            "acc":(tp.sum()/max(tot,1)).item()}

def train_scn(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    tl = build_rafdb_train(B)
    model = SCN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    crit = nn.CrossEntropyLoss()
    best_loss = float("inf"); best_state = None
    m1, m2 = 0.15, 0.2  # SCN hyperparams: rank regularization margin, relabel margin

    for ep in range(EPOCHS):
        model.train(); total_loss=0; nb=0
        for batch in tl:
            imgs, labels = batch
            if isinstance(imgs, list):
                rgb=torch.stack([pil_to_tensor01(center_crop_resize(p.convert("RGB"),FS)) for p in imgs]).to(DEVICE)
            else:
                rgb=imgs.to(DEVICE)
            labels=labels.to(DEVICE)
            logits, alphas = model(rgb, epoch=ep)
            ce_loss = crit(logits, labels)

            # Rank regularization
            high_mask = alphas.squeeze() >= alphas.squeeze().median()
            low_mask = ~high_mask
            if high_mask.sum()>0 and low_mask.sum()>0:
                loss_high = F.cross_entropy(logits[high_mask], labels[high_mask], reduction='mean')
                loss_low = F.cross_entropy(logits[low_mask], labels[low_mask], reduction='mean')
                rr_loss = F.relu(m1 - (loss_low - loss_high))
            else:
                rr_loss = 0.0

            loss = ce_loss + 0.1 * rr_loss
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); nb += 1

        avg = total_loss/max(nb,1); sched.step(avg)
        if avg < best_loss: best_loss=avg; best_state=deepcopy(model.state_dict())
        if (ep+1)%20==0: print(f"  Epoch {ep+1}/{EPOCHS}, loss={avg:.4f}")

    model.load_state_dict(best_state)
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
print("SCN 10-Seed Training")
print(f"New seeds: {NEW_SEEDS}")
print("="*60)

all_results = {}
res_path = OUT / "scn_results.json"
if res_path.exists():
    with open(res_path) as f: all_results = json.load(f)

for seed in NEW_SEEDS:
    key = f"SCN_s{seed}"
    if key in all_results:
        print(f"  [{key}] SKIP")
        continue
    t0=time.time()
    print(f"  [{key}] Training...")
    try:
        r = train_scn(seed)
        all_results[key] = r
        print(f"  [{key}] Mean={r['mean']:.4f} ({time.time()-t0:.0f}s)")
        with open(res_path,"w") as f: json.dump(all_results, f, indent=2)
    except Exception as e:
        print(f"  [{key}] ERROR: {e}")

vals=[v["mean"] for v in all_results.values() if "mean" in v]
print(f"\nSCN: {len(vals)} seeds, Mean={np.mean(vals):.4f} +/- {np.std(vals):.4f}")
