"""
MHAN and ViT-B/16 10-seed training.
MHAN bug: output is tuple (logits, attn) — must unpack.
ViT-S not available in torchvision 0.27 — replaced with ViT-B/16 10-seed.
"""
import sys, json, numpy as np, time
from copy import deepcopy
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import vit_b_16, ViT_B_16_Weights

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path("e:/scientific/小波/MHAN-code/MHAN-main")))
from src.dataset_registry import REGISTRY
from src.preprocess import center_crop_resize, pil_to_tensor01

DEVICE = torch.device("cuda")
B, EPOCHS, NC = 16, 40, 7
DATA_ROOT = Path("e:/scientific/小波/data")
OUT = _REPO / "runs" / "all_10seeds"
OUT.mkdir(parents=True, exist_ok=True)

NEW_SEEDS = [10, 89, 781, 999, 1337, 2026]
TARGETS = ["fer2013","affectnet","ckplus","jaffe"]
MEAN = [0.485,0.456,0.406]; STD = [0.229,0.224,0.225]

# ---- Data loaders ----
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
def evaluate(model, loader, mhan_mode=False, vit_mode=False):
    model.eval()
    tp=torch.zeros(NC,device=DEVICE); fp=torch.zeros(NC,device=DEVICE)
    fn=torch.zeros(NC,device=DEVICE); tot=0
    for batch in loader:
        imgs, labels = batch
        if isinstance(imgs, list):
            rgb=torch.stack([pil_to_tensor01(center_crop_resize(p.convert("RGB"),224)) for p in imgs]).to(DEVICE)
        else:
            rgb=imgs.to(DEVICE)
        labels=labels.to(DEVICE)
        if mhan_mode:
            rgb = F.interpolate(rgb, size=(112,112), mode='bilinear', align_corners=False)
        out = model(rgb)
        # MHAN returns tuple
        if isinstance(out, tuple): logits = out[0]
        else: logits = out
        pr=logits.argmax(dim=1); tot+=labels.size(0)
        for c in range(NC):
            tp[c]+=((pr==c)&(labels==c)).sum(); fp[c]+=((pr==c)&(labels!=c)).sum()
            fn[c]+=((pr!=c)&(labels==c)).sum()
    pre=tp/(tp+fp+1e-8); rec=tp/(tp+fn+1e-8)
    return {"macro_f1":(2*pre*rec/(pre+rec+1e-8)).mean().item(),
            "acc":(tp.sum()/max(tot,1)).item()}

# ---- MHAN ----
def train_mhan(seed):
    from networks.backbone import MHAN
    torch.manual_seed(seed); np.random.seed(seed)

    class MLoader:
        def __init__(self, loader): self.loader = loader
        def __iter__(self):
            for batch in self.loader:
                imgs, labels = batch
                if isinstance(imgs, list):
                    rgb = torch.stack([pil_to_tensor01(center_crop_resize(p.convert("RGB"),112)) for p in imgs]).to(DEVICE)
                else:
                    rgb = F.interpolate(imgs.to(DEVICE), size=(112,112), mode='bilinear', align_corners=False)
                yield rgb, labels.to(DEVICE)
        def __len__(self): return len(self.loader)

    tl = MLoader(build_rafdb_train(B))
    model = MHAN(num_class=NC, num_head=2, pretrained=False).to(DEVICE)
    mhan_pretrained = Path("e:/scientific/小波/MHAN-code/MHAN-main/pretrained/MFN_msceleb.pth")
    if mhan_pretrained.exists():
        pnet = torch.load(mhan_pretrained, map_location=DEVICE, weights_only=False)
        pfeat = nn.Sequential(*list(pnet.children())[:-4])
        model.features.load_state_dict(pfeat.state_dict(), strict=True)

    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    crit = nn.CrossEntropyLoss()
    best_loss=float("inf"); best_state=None
    for ep in range(EPOCHS):
        model.train(); total_loss=0; nb=0
        for rgb, labels in tl:
            opt.zero_grad()
            out = model(rgb)
            if isinstance(out, tuple): logits = out[0]  # FIX: unpack tuple
            else: logits = out
            loss = crit(logits, labels)
            loss.backward(); opt.step()
            total_loss += loss.item(); nb += 1
        avg=total_loss/max(nb,1); sched.step(avg)
        if avg<best_loss: best_loss=avg; best_state=deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    results={}
    for tgt in TARGETS:
        try:
            loader=build_target_loader(tgt, B)
            results[tgt]=round(evaluate(model, loader, mhan_mode=True)["macro_f1"],4)
        except Exception as e: results[tgt]=None
    results["mean"]=round(float(np.mean([v for v in results.values() if v])),4)
    return results

# ---- ViT-B/16 ----
class ViTFER(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.vit = vit_b_16(weights=weights)
        self.vit.heads = nn.Sequential(
            nn.Linear(768,256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256,NC))
    def forward(self, x): return self.vit(x)

def train_vit(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    tl = build_rafdb_train(B)
    model = ViTFER().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    crit = nn.CrossEntropyLoss()
    best_loss=float("inf"); best_state=None
    for ep in range(EPOCHS):
        model.train(); total_loss=0; nb=0
        for batch in tl:
            imgs, labels = batch
            if isinstance(imgs, list):
                rgb=torch.stack([pil_to_tensor01(center_crop_resize(p.convert("RGB"),224)) for p in imgs]).to(DEVICE)
            else: rgb=imgs.to(DEVICE)
            labels=labels.to(DEVICE)
            opt.zero_grad(); loss=crit(model(rgb), labels)
            loss.backward(); opt.step()
            total_loss+=loss.item(); nb+=1
        avg=total_loss/max(nb,1); sched.step(avg)
        if avg<best_loss: best_loss=avg; best_state=deepcopy(model.state_dict())
        if (ep+1)%20==0: print(f"  Epoch {ep+1}/{EPOCHS}, loss={avg:.4f}")
    model.load_state_dict(best_state)
    results={}
    for tgt in TARGETS:
        try:
            loader=build_target_loader(tgt, B)
            results[tgt]=round(evaluate(model, loader)["macro_f1"],4)
        except Exception as e: results[tgt]=None
    results["mean"]=round(float(np.mean([v for v in results.values() if v])),4)
    return results

# ---- Main ----
print("="*60)
print("MHAN + ViT-B/16 Fix & Re-run")
print("="*60)

all_results = {}
res_path = OUT / "all_results.json"
if res_path.exists():
    with open(res_path) as f: all_results = json.load(f)

for method_name, train_fn, prefix in [
    ("MHAN", train_mhan, "MHAN"),
    ("ViT-B", train_vit, "ViT-B"),
]:
    print(f"\n{'='*60}")
    print(f"METHOD: {method_name}")
    print(f"{'='*60}")
    for seed in NEW_SEEDS:
        key = f"{prefix}_s{seed}"
        if key in all_results and "mean" in all_results[key] and all_results[key]["mean"] > 0.01:
            print(f"  [{key}] SKIP (valid result exists)")
            continue
        # Remove any stale failed entry
        if key in all_results: del all_results[key]
        t0=time.time()
        print(f"  [{key}] Training...")
        try:
            r = train_fn(seed)
            all_results[key] = r
            print(f"  [{key}] Mean={r['mean']:.4f} ({time.time()-t0:.0f}s)")
            with open(res_path,"w") as f: json.dump(all_results, f, indent=2)
        except Exception as e:
            print(f"  [{key}] ERROR: {e}")
            import traceback; traceback.print_exc()
    torch.cuda.empty_cache()

# Summary
vals_m=[v["mean"] for k,v in all_results.items() if k.startswith("MHAN") and "mean" in v]
vals_v=[v["mean"] for k,v in all_results.items() if k.startswith("ViT-B") and "mean" in v]
print(f"\nMHAN: {len(vals_m)} seeds, Mean={np.mean(vals_m):.4f} +/- {np.std(vals_m,ddof=1):.4f}" if len(vals_m)>1 else f"\nMHAN: {len(vals_m)} seeds")
print(f"ViT-B: {len(vals_v)} seeds, Mean={np.mean(vals_v):.4f} +/- {np.std(vals_v,ddof=1):.4f}" if len(vals_v)>1 else f"ViT-B: {len(vals_v)} seeds")
print("\nDone!")
