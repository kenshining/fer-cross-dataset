"""
10-seed training for primary comparison methods (RAF-DB source).
Seeds: 42, 123 (existing) + 456, 789 (done/running) + 0, 1, 2, 3, 4, 5 (new).
Each method->4 targets: FER2013, AffectNet, CK+, JAFFE.
"""
import sys, json, numpy as np, os, time
from copy import deepcopy
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms.functional as TF

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path("e:/scientific/小波/MHAN-code/MHAN-main")))
from src.dataset_registry import REGISTRY
from src.preprocess import center_crop_resize, pil_to_tensor01

DEVICE = torch.device("cuda")
B, LR, EPOCHS, NC, FS = 16, 1e-3, 40, 7, 224
DATA_ROOT = Path("e:/scientific/小波/data")
OUT = _REPO / "runs" / "all_10seeds"
OUT.mkdir(parents=True, exist_ok=True)

# Final seed set (10 diverse ML-standard seeds)
ALL_SEEDS = [10, 42, 89, 123, 456, 781, 789, 999, 1337, 2026]
NEW_SEEDS = [10, 89, 781, 999, 1337, 2026]  # only run these (others exist/running)

TARGETS = ["fer2013", "affectnet", "ckplus", "jaffe"]
MEAN = [0.485, 0.456, 0.406]; STD = [0.229, 0.224, 0.225]

print("="*60)
print("Comprehensive 10-Seed Training")
print(f"Seeds: {ALL_SEEDS}")
print(f"New to run: {NEW_SEEDS}")
print(f"Targets: {TARGETS}")
print("="*60)

# ===================================================================
# Base ResNet-18 model
# ===================================================================
class RN(nn.Module):
    def __init__(self):
        super().__init__()
        b = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.e = nn.Sequential(*list(b.children())[:-1])
        self.c = nn.Sequential(nn.Flatten(), nn.Linear(512,256), nn.ReLU(),
                               nn.Dropout(0.5), nn.Linear(256,NC))
    def forward(self,x): return self.c(self.e(x))

# ===================================================================
# Data loaders
# ===================================================================
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

# ===================================================================
# Eval
# ===================================================================
@torch.no_grad()
def evaluate(model, loader, mhan_mode=False):
    model.eval()
    tp=torch.zeros(NC,device=DEVICE); fp=torch.zeros(NC,device=DEVICE)
    fn=torch.zeros(NC,device=DEVICE); tot=0
    for bd, lb in loader:
        lb=lb.to(DEVICE)
        if isinstance(bd, list):
            rgb=torch.stack([pil_to_tensor01(center_crop_resize(p.convert("RGB"),FS)) for p in bd]).to(DEVICE)
        else:
            rgb=bd.to(DEVICE)
        if mhan_mode:
            from networks.backbone import MHAN
            rgb = F.interpolate(rgb, size=(112,112), mode='bilinear', align_corners=False)
        pr=model(rgb).argmax(dim=1); tot+=lb.size(0)
        for c in range(NC):
            tp[c]+=((pr==c)&(lb==c)).sum(); fp[c]+=((pr==c)&(lb!=c)).sum()
            fn[c]+=((pr!=c)&(lb==c)).sum()
    pre=tp/(tp+fp+1e-8); rec=tp/(tp+fn+1e-8)
    return {"macro_f1": (2*pre*rec/(pre+rec+1e-8)).mean().item(),
            "acc": (tp.sum()/max(tot,1)).item()}

# ===================================================================
# Augmentation helpers
# ===================================================================
def randaug(rgb):
    B=rgb.size(0); out=[]
    for i in range(B):
        img=(rgb[i]*255).byte()
        if np.random.random()>.5: img=TF.adjust_brightness(img,float(.8+np.random.random()*.4))
        if np.random.random()>.5: img=TF.adjust_contrast(img,float(.8+np.random.random()*.4))
        if np.random.random()>.5: img=TF.adjust_sharpness(img,float(1.+np.random.random()))
        out.append(img.float()/255.)
    return torch.stack(out)

def mixup(rgb, alpha=0.2):
    B=rgb.size(0); lam=np.random.beta(alpha,alpha) if alpha>0 else 1.
    idx=torch.randperm(B,device=rgb.device)
    return lam*rgb+(1-lam)*rgb[idx]

def mixup_crit(crit, pred, y_a, y_b, lam):
    return lam*crit(pred,y_a)+(1-lam)*crit(pred,y_b)

def preprocess_batch(imgs):
    if isinstance(imgs, list):
        return torch.stack([pil_to_tensor01(center_crop_resize(p.convert("RGB"),FS)) for p in imgs]).to(DEVICE)
    return imgs.to(DEVICE)

# ===================================================================
# Training routines
# ===================================================================
def train_resnet(seed, epochs=EPOCHS):
    torch.manual_seed(seed); np.random.seed(seed)
    tl = build_rafdb_train(B)
    model = RN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    crit = nn.CrossEntropyLoss()
    best_loss = float("inf"); best_state = None
    for ep in range(epochs):
        model.train(); total_loss=0; nb=0
        for batch in tl:
            rgb, labels = preprocess_batch(batch[0]), batch[1].to(DEVICE)
            opt.zero_grad(); loss = crit(model(rgb), labels)
            loss.backward(); opt.step()
            total_loss += loss.item(); nb += 1
        avg = total_loss/max(nb,1); sched.step(avg)
        if avg < best_loss: best_loss=avg; best_state=deepcopy(model.state_dict())
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

def train_with_aug(seed, aug_fn=None, mixup_mode=False, label="aug"):
    torch.manual_seed(seed); np.random.seed(seed)
    tl = build_rafdb_train(B)
    model = RN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    crit = nn.CrossEntropyLoss()
    best_loss = float("inf"); best_state = None
    for ep in range(EPOCHS):
        model.train(); total_loss=0; nb=0
        for batch in tl:
            rgb, labels = preprocess_batch(batch[0]), batch[1].to(DEVICE)
            if aug_fn: rgb = aug_fn(rgb)
            if mixup_mode:
                lam=np.random.beta(.2,.2) if .2>0 else 1.
                idx=torch.randperm(rgb.size(0),device=DEVICE)
                mixed=lam*rgb+(1-lam)*rgb[idx]
                opt.zero_grad(); loss=mixup_crit(crit,model(mixed),labels,labels[idx],lam)
            else:
                opt.zero_grad(); loss = crit(model(rgb), labels)
            loss.backward(); opt.step()
            total_loss += loss.item(); nb += 1
        avg = total_loss/max(nb,1); sched.step(avg)
        if avg < best_loss: best_loss=avg; best_state=deepcopy(model.state_dict())
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

def train_mhan(seed):
    from networks.backbone import MHAN
    torch.manual_seed(seed); np.random.seed(seed)
    # MHAN uses 112x112
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
    best_loss = float("inf"); best_state = None
    for ep in range(EPOCHS):
        model.train(); total_loss=0; nb=0
        for rgb, labels in tl:
            opt.zero_grad(); loss = crit(model(rgb), labels)
            loss.backward(); opt.step()
            total_loss += loss.item(); nb += 1
        avg = total_loss/max(nb,1); sched.step(avg)
        if avg < best_loss: best_loss=avg; best_state=deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    results = {}
    for tgt in TARGETS:
        try:
            loader = build_target_loader(tgt, B)
            results[tgt] = round(evaluate(model, loader, mhan_mode=True)["macro_f1"], 4)
        except Exception as e:
            results[tgt] = None; print(f"  [ERR] {tgt}: {e}")
    results["mean"] = round(float(np.mean([v for v in results.values() if v])), 4)
    return results

def train_vit(seed, variant="S"):
    from torchvision.models import vit_b_16, vit_s_16, ViT_B_16_Weights, ViT_S_16_Weights
    torch.manual_seed(seed); np.random.seed(seed)
    if variant == "S":
        weights = ViT_S_16_Weights.IMAGENET1K_V1
        vit = vit_s_16(weights=weights)
        dim = 384
    else:
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        vit = vit_b_16(weights=weights)
        dim = 768
    vit.heads = nn.Sequential(nn.Linear(dim,256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256,NC))
    model = vit.to(DEVICE)
    tl = build_rafdb_train(B)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    crit = nn.CrossEntropyLoss()
    best_loss = float("inf"); best_state = None
    for ep in range(EPOCHS):
        model.train(); total_loss=0; nb=0
        for batch in tl:
            rgb, labels = preprocess_batch(batch[0]), batch[1].to(DEVICE)
            opt.zero_grad(); loss = crit(model(rgb), labels)
            loss.backward(); opt.step()
            total_loss += loss.item(); nb += 1
        avg = total_loss/max(nb,1); sched.step(avg)
        if avg < best_loss: best_loss=avg; best_state=deepcopy(model.state_dict())
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

# ===================================================================
# Main: run all new seeds for all methods
# ===================================================================
all_results = {}

# Queue: methods to run with new seeds
methods = [
    ("ResNet18",       lambda s: train_resnet(s)),
    ("RandAugment",    lambda s: train_with_aug(s, aug_fn=randaug, label="RandAug")),
    ("MixUp",          lambda s: train_with_aug(s, mixup_mode=True, label="MixUp")),
    ("MHAN",           lambda s: train_mhan(s)),
    ("ViT-S",          lambda s: train_vit(s, "S")),
    ("ViT-B",          lambda s: train_vit(s, "B")),
    # SCN and RUL require their specific model architectures - skip for now,
    # add after verifying SCN/RUL training scripts work with updated seeds
]

# Load existing results if available
existing_path = OUT / "all_results.json"
if existing_path.exists():
    with open(existing_path) as f:
        all_results = json.load(f)
    print(f"Loaded {len(all_results)} existing results")

for method_name, train_fn in methods:
    print(f"\n{'='*60}")
    print(f"METHOD: {method_name}")
    print(f"{'='*60}")

    for seed in NEW_SEEDS:
        key = f"{method_name}_s{seed}"
        if key in all_results:
            print(f"  [{key}] SKIP (already done)")
            continue

        t0 = time.time()
        print(f"  [{key}] Training...")
        try:
            results = train_fn(seed)
            all_results[key] = results
            elapsed = time.time() - t0
            print(f"  [{key}] Mean={results['mean']:.4f} ({elapsed:.0f}s)")

            # Save incrementally (crash-safe)
            with open(existing_path, "w") as f:
                json.dump(all_results, f, indent=2)
        except Exception as e:
            print(f"  [{key}] ERROR: {e}")
            import traceback; traceback.print_exc()

    torch.cuda.empty_cache()
    print(f"  {method_name}: {len([k for k in all_results if k.startswith(method_name)])} seeds complete")

# ===================================================================
# Summary
# ===================================================================
print(f"\n{'='*60}")
print(f"TOTAL: {len(all_results)} runs completed")
print(f"{'='*60}")

# Per-method mean ± std
from collections import defaultdict
method_stats = defaultdict(list)
for key, res in all_results.items():
    method = "_".join(key.split("_")[:-1])  # remove _sXXX
    if "mean" in res:
        method_stats[method].append(res["mean"])

print(f"\n{'Method':<20} {'Seeds':>6} {'Mean Macro-F1':>14} {'Std':>8}")
print("-"*50)
for method, vals in sorted(method_stats.items()):
    print(f"{method:<20} {len(vals):>6} {np.mean(vals):>14.4f} {np.std(vals):>8.4f}")

with open(existing_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nFinal results saved to {existing_path}")
print("All experiments complete!")
