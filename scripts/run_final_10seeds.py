"""
Complete 10-seed training for all baseline methods (RAF-DB source).
Supports incremental execution with crash-safe JSON checkpointing.
"""
import sys, json, numpy as np, time
from copy import deepcopy
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights, vit_b_16, ViT_B_16_Weights
import torchvision.transforms.functional as TF

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path("e:/scientific/小波/MHAN-code/MHAN-main")))
from src.dataset_registry import REGISTRY
from src.preprocess import center_crop_resize, pil_to_tensor01
from src.scn_model import SelfAttentionWeighting

DEVICE = torch.device("cuda")
B, NC, FS = 16, 7, 224
DATA_ROOT = Path("e:/scientific/小波/data")
OUT = _REPO / "runs" / "all_10seeds"
OUT.mkdir(parents=True, exist_ok=True)

ALL10 = [10, 42, 89, 123, 456, 781, 789, 999, 1337, 2026]
TARGETS = ["fer2013","affectnet","ckplus","jaffe"]

# ====== DATA ======
def build_rafdb(bs):
    ds=REGISTRY["rafdb"]["dataset_cls"](DATA_ROOT/"RAF-DB",split="train")
    return DataLoader(ds,batch_size=bs,shuffle=True,collate_fn=REGISTRY["rafdb"]["collate_fn"],num_workers=0)

def build_target(name,bs):
    if name=="fer2013":
        import tempfile,csv
        from src.dataset_fer2013 import FER2013Dataset,fer2013_collate_fn
        cp=DATA_ROOT/"Fer2013"/"fer2013"/"fer2013.csv"
        tmp=tempfile.NamedTemporaryFile(mode="w",suffix=".csv",delete=False,encoding="utf-8")
        tmp.write("emotion,pixels\n")
        with open(cp) as f:
            for line in f:
                if "PublicTest" in line:
                    p=line.strip().split(",",2)
                    if len(p)>=2: tmp.write(f"{p[0]},{p[1]}\n")
        tmp.close()
        return DataLoader(FER2013Dataset(Path(tmp.name)),batch_size=bs,shuffle=False,collate_fn=fer2013_collate_fn,num_workers=0)
    elif name=="affectnet":
        ds=REGISTRY["affectnet"]["dataset_cls"](DATA_ROOT/"AffectNet",split="val")
        return DataLoader(ds,batch_size=bs,shuffle=False,collate_fn=REGISTRY["affectnet"]["collate_fn"],num_workers=0)
    elif name=="ckplus":
        ds=REGISTRY["ckplus"]["dataset_cls"](DATA_ROOT/"CK+")
        return DataLoader(ds,batch_size=bs,shuffle=False,collate_fn=REGISTRY["ckplus"]["collate_fn"],num_workers=0)
    elif name=="jaffe":
        ds=REGISTRY["jaffe"]["dataset_cls"](DATA_ROOT/"Jaffe")
        return DataLoader(ds,batch_size=bs,shuffle=False,collate_fn=REGISTRY["jaffe"]["collate_fn"],num_workers=0)

def proc_batch(imgs):
    if isinstance(imgs,list):
        return torch.stack([pil_to_tensor01(center_crop_resize(p.convert("RGB"),FS)) for p in imgs]).to(DEVICE)
    return imgs.to(DEVICE)

@torch.no_grad()
def ev(model,loader,mhan=False):
    model.eval();tp=torch.zeros(NC,device=DEVICE);fp=torch.zeros(NC,device=DEVICE)
    fn=torch.zeros(NC,device=DEVICE);tot=0
    for bd,lb in loader:
        lb=lb.to(DEVICE);rgb=proc_batch(bd)
        if mhan: rgb=F.interpolate(rgb,size=(112,112),mode="bilinear")
        out=model(rgb)
        if isinstance(out,tuple): logits=out[0]
        else: logits=out
        pr=logits.argmax(dim=1);tot+=lb.size(0)
        for c in range(NC):
            tp[c]+=((pr==c)&(lb==c)).sum();fp[c]+=((pr==c)&(lb!=c)).sum();fn[c]+=((pr!=c)&(lb==c)).sum()
    pre=tp/(tp+fp+1e-8);rec=tp/(tp+fn+1e-8)
    return {"macro_f1":(2*pre*rec/(pre+rec+1e-8)).mean().item(),"acc":(tp.sum()/max(tot,1)).item()}

# ====== MODELS ======
class RN(nn.Module):
    def __init__(s):
        super().__init__();b=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        s.e=nn.Sequential(*list(b.children())[:-1])
        s.c=nn.Sequential(nn.Flatten(),nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.5),nn.Linear(256,NC))
    def forward(s,x): return s.c(s.e(x))

class SCNModel(nn.Module):
    def __init__(s):
        super().__init__();b=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        s.encoder=nn.Sequential(*list(b.children())[:-1])
        s.alpha=SelfAttentionWeighting(512)
        s.cls=nn.Sequential(nn.Flatten(),nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.5),nn.Linear(256,NC))
    def forward(s,x): f=s.encoder(x).view(x.size(0),-1);a=s.alpha(f);return s.cls(f),a

class RULModel(nn.Module):
    def __init__(s):
        super().__init__();b=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        s.encoder=nn.Sequential(*list(b.children())[:-1])
        s.cls=nn.Sequential(nn.Flatten(),nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.5),nn.Linear(256,NC))
        s.u_head=nn.Linear(512,1)
    def forward(s,x): f=s.encoder(x).view(x.size(0),-1);return s.cls(f),torch.sigmoid(s.u_head(f))

class ViTB(nn.Module):
    def __init__(s):
        super().__init__();w=ViT_B_16_Weights.IMAGENET1K_V1;s.vit=vit_b_16(weights=w)
        s.vit.heads=nn.Sequential(nn.Linear(768,256),nn.ReLU(),nn.Dropout(0.3),nn.Linear(256,NC))
    def forward(s,x): return s.vit(x)

# ====== TRAINERS ======
def train_std(model,seed,epochs=40,lr=1e-3):
    torch.manual_seed(seed);np.random.seed(seed)
    tl=build_rafdb(B);opt=torch.optim.Adam(model.parameters(),lr=lr)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=0.5,patience=5)
    crit=nn.CrossEntropyLoss();best_loss=float("inf");best_state=None
    for ep in range(epochs):
        model.train();tot=0;nb=0
        for batch in tl:
            rgb,lb=proc_batch(batch[0]),batch[1].to(DEVICE)
            opt.zero_grad();loss=crit(model(rgb),lb);loss.backward();opt.step();tot+=loss.item();nb+=1
        avg=tot/max(nb,1);sched.step(avg)
        if avg<best_loss: best_loss=avg;best_state=deepcopy(model.state_dict())
    model.load_state_dict(best_state);return model

def train_scn(seed):
    torch.manual_seed(seed);np.random.seed(seed)
    tl=build_rafdb(B);model=SCNModel().to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=0.5,patience=5)
    crit=nn.CrossEntropyLoss();best_loss=float("inf");best_state=None
    for ep in range(40):
        model.train();tot=0;nb=0
        for batch in tl:
            rgb,lb=proc_batch(batch[0]),batch[1].to(DEVICE)
            logits,alphas=model(rgb);ce=crit(logits,lb)
            a=alphas.squeeze();hi=a>=a.median();lo=~hi
            rr=F.relu(0.15-((F.cross_entropy(logits[lo],lb[lo],reduction='mean') if lo.sum()>0 else 0)-(F.cross_entropy(logits[hi],lb[hi],reduction='mean') if hi.sum()>0 else 0))) if hi.sum()>0 and lo.sum()>0 else 0
            loss=ce+0.1*rr;opt.zero_grad();loss.backward();opt.step();tot+=loss.item();nb+=1
        avg=tot/max(nb,1);sched.step(avg)
        if avg<best_loss: best_loss=avg;best_state=deepcopy(model.state_dict())
    model.load_state_dict(best_state);return model

def train_rul(seed):
    torch.manual_seed(seed);np.random.seed(seed)
    tl=build_rafdb(B);model=RULModel().to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=0.5,patience=5)
    crit=nn.CrossEntropyLoss();best_loss=float("inf");best_state=None
    for ep in range(40):
        model.train();tot=0;nb=0
        for batch in tl:
            rgb,lb=proc_batch(batch[0]),batch[1].to(DEVICE)
            f=model.encoder(rgb).view(rgb.size(0),-1)
            idx=torch.randperm(rgb.size(0),device=DEVICE);lam=np.random.beta(0.5,0.5)
            fm=lam*f+(1-lam)*f[idx];u=torch.sigmoid(model.u_head(fm));logits=model.cls(fm)
            loss=lam*crit(logits,lb)+(1-lam)*crit(logits,lb[idx])+0.1*u.mean()
            opt.zero_grad();loss.backward();opt.step();tot+=loss.item();nb+=1
        avg=tot/max(nb,1);sched.step(avg)
        if avg<best_loss: best_loss=avg;best_state=deepcopy(model.state_dict())
    model.load_state_dict(best_state);return model

def train_randaug(seed):
    def ra(rgb):
        B=rgb.size(0);out=[]
        for i in range(B):
            img=(rgb[i]*255).byte()
            if np.random.random()>.5: img=TF.adjust_brightness(img,float(.8+np.random.random()*.4))
            if np.random.random()>.5: img=TF.adjust_contrast(img,float(.8+np.random.random()*.4))
            if np.random.random()>.5: img=TF.adjust_sharpness(img,float(1.+np.random.random()))
            out.append(img.float()/255.)
        return torch.stack(out)
    torch.manual_seed(seed);np.random.seed(seed)
    tl=build_rafdb(B);model=RN().to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=0.5,patience=5)
    crit=nn.CrossEntropyLoss();best_loss=float("inf");best_state=None
    for ep in range(40):
        model.train();tot=0;nb=0
        for batch in tl:
            rgb,lb=ra(proc_batch(batch[0])),batch[1].to(DEVICE)
            opt.zero_grad();loss=crit(model(rgb),lb);loss.backward();opt.step();tot+=loss.item();nb+=1
        avg=tot/max(nb,1);sched.step(avg)
        if avg<best_loss: best_loss=avg;best_state=deepcopy(model.state_dict())
    model.load_state_dict(best_state);return model

def train_mixup(seed):
    torch.manual_seed(seed);np.random.seed(seed)
    tl=build_rafdb(B);model=RN().to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=0.5,patience=5)
    crit=nn.CrossEntropyLoss();best_loss=float("inf");best_state=None
    for ep in range(40):
        model.train();tot=0;nb=0
        for batch in tl:
            rgb,lb=proc_batch(batch[0]),batch[1].to(DEVICE)
            lam=np.random.beta(.2,.2);idx=torch.randperm(rgb.size(0),device=DEVICE)
            mixed=lam*rgb+(1-lam)*rgb[idx]
            opt.zero_grad();loss=lam*crit(model(mixed),lb)+(1-lam)*crit(model(mixed),lb[idx])
            loss.backward();opt.step();tot+=loss.item();nb+=1
        avg=tot/max(nb,1);sched.step(avg)
        if avg<best_loss: best_loss=avg;best_state=deepcopy(model.state_dict())
    model.load_state_dict(best_state);return model

def train_mhan(seed):
    from networks.backbone import MHAN
    torch.manual_seed(seed);np.random.seed(seed)
    class ML:
        def __init__(s,l): s.l=l
        def __iter__(s):
            for b in s.l:
                imgs,lb=b
                if isinstance(imgs,list): rgb=torch.stack([pil_to_tensor01(center_crop_resize(p.convert("RGB"),112)) for p in imgs]).to(DEVICE)
                else: rgb=F.interpolate(imgs.to(DEVICE),size=(112,112),mode="bilinear")
                yield rgb,lb.to(DEVICE)
        def __len__(s): return len(s.l)
    tl=ML(build_rafdb(B));model=MHAN(num_class=NC,num_head=2,pretrained=False).to(DEVICE)
    pretrained=Path("e:/scientific/小波/MHAN-code/MHAN-main/pretrained/MFN_msceleb.pth")
    if pretrained.exists():
        pnet=torch.load(pretrained,map_location=DEVICE,weights_only=False)
        pfeat=nn.Sequential(*list(pnet.children())[:-4])
        model.features.load_state_dict(pfeat.state_dict(),strict=True)
    opt=torch.optim.Adam(model.parameters(),lr=5e-4)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=0.5,patience=5)
    crit=nn.CrossEntropyLoss();best_loss=float("inf");best_state=None
    for ep in range(40):
        model.train();tot=0;nb=0
        for rgb,lb in tl:
            opt.zero_grad();out=model(rgb);logits=out[0] if isinstance(out,tuple) else out
            loss=crit(logits,lb);loss.backward();opt.step();tot+=loss.item();nb+=1
        avg=tot/max(nb,1);sched.step(avg)
        if avg<best_loss: best_loss=avg;best_state=deepcopy(model.state_dict())
    model.load_state_dict(best_state);return model

def train_vit(seed):
    model=ViTB().to(DEVICE);return train_std(model,seed,lr=1e-4)

def train_swad(seed):
    """SWAD: dense-to-sparse SWA."""
    torch.manual_seed(seed);np.random.seed(seed)
    tl=build_rafdb(B);model=RN().to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=0.5,patience=5)
    crit=nn.CrossEntropyLoss();ckpts=[]
    for ep in range(40):
        model.train();tot=0;nb=0
        for batch in tl:
            rgb,lb=proc_batch(batch[0]),batch[1].to(DEVICE)
            opt.zero_grad();loss=crit(model(rgb),lb);loss.backward();opt.step();tot+=loss.item();nb+=1
        avg=tot/max(nb,1);sched.step(avg)
        if ep>=20: ckpts.append(deepcopy(model.state_dict()))
    # Dense-to-sparse: keep every 2nd checkpoint
    keep=ckpts[::2];avg_state=deepcopy(keep[0])
    for s in keep[1:]:
        for k in avg_state: avg_state[k]+=s[k]
    for k in avg_state:
	        if avg_state[k].dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
	            avg_state[k] /= len(keep)
    model.load_state_dict(avg_state);return model

# ====== MAIN ======
all_results={}
res_path=OUT/"all_results.json"
if res_path.exists():
    with open(res_path) as f: all_results=json.load(f)

QUEUE=[
    ("RandAug", [456,789], train_randaug),
    ("MixUp",   [456,789], train_mixup),
    ("MHAN",    [456,789], train_mhan),
    ("ViT-B",   ALL10,    train_vit),
    ("SCN",     ALL10,    train_scn),
    ("RUL",     ALL10,    train_rul),
    ("SWAD",    ALL10,    train_swad),
]

print("="*60)
print("FINAL 10-Seed Training: All Methods")
print(f"Total queue: {sum(len(seeds) for _,seeds,_ in QUEUE)} runs")
print("="*60)

for method, seeds, train_fn in QUEUE:
    print(f"\n{'='*60}")
    print(f"METHOD: {method} ({len(seeds)} target seeds)")
    print(f"{'='*60}")
    for seed in seeds:
        key=f"{method}_s{seed}"
        if key in all_results and "mean" in all_results.get(key,{}) and all_results[key].get("mean",0)>0.01:
            continue
        if key in all_results: del all_results[key]  # remove stale
        t0=time.time();print(f"  [{key}] Training...")
        try:
            model=train_fn(seed);model.eval()
            r={}
            for tgt in TARGETS:
                try:
                    loader=build_target(tgt,B)
                    m=ev(model,loader,mhan=(method=="MHAN"))
                    r[tgt]=round(m["macro_f1"],4)
                except Exception as e:
                    r[tgt]=None;print(f"    [ERR] {tgt}: {e}")
            r["mean"]=round(float(np.mean([v for v in r.values() if v])),4)
            all_results[key]=r
            print(f"  [{key}] Mean={r['mean']:.4f} ({time.time()-t0:.0f}s)")
            with open(res_path,"w") as f: json.dump(all_results,f,indent=2)
        except Exception as e:
            print(f"  [{key}] ERROR: {e}")
            import traceback;traceback.print_exc()
    torch.cuda.empty_cache()

# Final summary
print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"{'='*60}")
from collections import defaultdict
counts=defaultdict(list)
for k,v in all_results.items():
    if "mean" in v:
        m=k.rsplit("_s",1)[0];counts[m].append(v["mean"])
for m,vals in sorted(counts.items()):
    print(f"{m:<15} {len(vals):>3} seeds  Mean={np.mean(vals):.4f} +/- {np.std(vals,ddof=1):.4f}" if len(vals)>1 else f"{m:<15} {len(vals):>3} seeds  Mean={np.mean(vals):.4f}")
print("\nALL DONE!")
