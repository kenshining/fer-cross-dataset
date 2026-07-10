"""
Post-hoc analysis of SCN self-attention weights.
Extends the original alpha analysis with median-split accuracy comparison,
point-biserial correlation, and partial correlation controlling for confidence.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
from scipy import stats

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

OUT_DIR = _REPO / "runs" / "scn_alpha_posthoc"
os.makedirs(OUT_DIR, exist_ok=True)

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# ========== Imports + Model ==========
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image

DATA_ROOT = Path("e:/scientific/小波/data")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 7
MEAN = [0.485, 0.456, 0.406]; STD = [0.229, 0.224, 0.225]

from src.scn_model import SelfAttentionWeighting

class SCNExtractor(nn.Module):
    def __init__(self, ckpt_path: str):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.alpha_module = SelfAttentionWeighting(512)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(256, NUM_CLASSES),
        )
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model_state = ckpt["model"]
        enc_state = {k.replace("encoder.",""): v for k,v in model_state.items() if k.startswith("encoder.")}
        self.encoder.load_state_dict(enc_state, strict=False)
        alpha_state = {k.replace("alpha_module.",""): v for k,v in model_state.items() if k.startswith("alpha_module.")}
        self.alpha_module.load_state_dict(alpha_state, strict=True)
        cls_state = {k.replace("classifier.",""): v for k,v in model_state.items() if k.startswith("classifier.")}
        if cls_state: self.classifier.load_state_dict(cls_state, strict=True)
        self.eval().to(DEVICE)

    @torch.no_grad()
    def extract(self, loader):
        alphas, labels_all, confs_all, correct_all = [], [], [], []
        for batch_data, labels in loader:
            rgb = batch_data.to(DEVICE)
            rgb = transforms.Normalize(mean=MEAN, std=STD)(rgb)
            f = self.encoder(rgb).view(rgb.size(0), -1)
            a = self.alpha_module(f).squeeze(-1)
            logits = self.classifier(f)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            alphas.append(a.cpu().numpy())
            lb = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
            pr = pred.cpu().numpy()
            labels_all.append(lb)
            confs_all.append(conf.cpu().numpy())
            correct_all.append((pr == lb).astype(np.float32))
        return (np.concatenate(alphas), np.concatenate(labels_all),
                np.concatenate(confs_all), np.concatenate(correct_all))

def build_raf_loader(split="test"):
    from src.dataset_registry import REGISTRY
    ds = REGISTRY["rafdb"]["dataset_cls"](DATA_ROOT / "RAF-DB", split=split)
    class W(torch.utils.data.Dataset):
        def __init__(s): s.ds = ds
        def __len__(s): return len(s.ds)
        def __getitem__(s, i):
            p, l = s.ds[i]
            p = p.convert("RGB") if hasattr(p, "convert") else p
            t = transforms.ToTensor()(transforms.Resize((224,224))(p))
            return t, l
    return DataLoader(W(), batch_size=64, shuffle=False)

def build_fer_loader(n=500):
    import csv
    csv_path = DATA_ROOT / "Fer2013" / "fer2013" / "fer2013.csv"
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("Usage","") == "PublicTest":
                rows.append((row["pixels"], int(row["emotion"])))
    np.random.seed(42); np.random.shuffle(rows); rows = rows[:n]
    tensors, labels = [], []
    for pixels_str, label in rows:
        pix = np.fromstring(pixels_str, sep=" ", dtype=np.uint8)
        img = pix.reshape(48,48); img = np.stack([img]*3, axis=-1)
        pil = Image.fromarray(img).resize((224,224), Image.BILINEAR)
        tensors.append(transforms.ToTensor()(pil)); labels.append(label)
    return DataLoader(torch.utils.data.TensorDataset(torch.stack(tensors), torch.tensor(labels)),
                      batch_size=64, shuffle=False)

# ---- Find checkpoint ----
scn_dir = _REPO / "runs" / "scn_baseline"
ckpt_paths = list(scn_dir.glob("scn_rafdb_seed*/best.pt"))
if not ckpt_paths:
    print(f"ERROR: No SCN checkpoints found in {scn_dir}")
    sys.exit(1)

print(f"\nFound {len(ckpt_paths)} SCN checkpoint(s)")

all_results = {}

for ckpt_path in ckpt_paths:
    seed = ckpt_path.parent.name
    print(f"\n--- Seed: {seed} ---")

    extractor = SCNExtractor(str(ckpt_path))

    # In-domain: RAF-DB test
    print("  Extracting in-domain (RAF-DB)...")
    raf_loader = build_raf_loader("test")
    alphas_id, labels_id, confs_id, correct_id = extractor.extract(raf_loader)

    # Cross-domain: FER2013 test
    print("  Extracting cross-domain (FER2013)...")
    fer_loader = build_fer_loader(500)
    alphas_cd, labels_cd, confs_cd, correct_cd = extractor.extract(fer_loader)

    # ---- Analysis 1: Alpha median split accuracy ----
    median_alpha = np.median(alphas_cd)
    high_mask = alphas_cd >= median_alpha
    low_mask = alphas_cd < median_alpha

    acc_high = correct_cd[high_mask].mean()
    acc_low = correct_cd[low_mask].mean()

    print(f"  [Split] High-alpha acc: {acc_high:.4f}, Low-alpha acc: {acc_low:.4f}")
    print(f"  [Split] Delta: {acc_high - acc_low:+.4f}")

    # Per-class split
    class_split = {}
    for c in range(NUM_CLASSES):
        c_mask = labels_cd == c
        if c_mask.sum() < 5: continue
        median_c = np.median(alphas_cd[c_mask])
        high_c = (alphas_cd[c_mask] >= median_c)
        low_c = (alphas_cd[c_mask] < median_c)
        class_split[EMOTIONS[c]] = {
            "n": int(c_mask.sum()),
            "acc_high": float(correct_cd[c_mask][high_c].mean()),
            "acc_low": float(correct_cd[c_mask][low_c].mean()),
            "delta": float(correct_cd[c_mask][high_c].mean() - correct_cd[c_mask][low_c].mean()),
            "mean_alpha": float(alphas_cd[c_mask].mean()),
        }

    # ---- Analysis 2: Point-biserial correlation ----
    # alpha vs correctness (across all cross-domain samples)
    r_pb, p_pb = stats.pointbiserialr(correct_cd, alphas_cd)
    print(f"  [PB-r] alpha x correctness: r={r_pb:.4f}, p={p_pb:.4f}")

    # ---- Analysis 3: Partial correlation ----
    # alpha vs correctness, controlling for confidence
    from scipy.stats import pearsonr
    def partial_corr(x, y, z):
        """Partial correlation r_{xy.z}"""
        r_xy, _ = pearsonr(x, y)
        r_xz, _ = pearsonr(x, z)
        r_yz, _ = pearsonr(y, z)
        num = r_xy - r_xz * r_yz
        den = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
        return num / max(den, 1e-8)

    r_partial = partial_corr(alphas_cd, correct_cd, confs_cd)
    print(f"  [Partial] alpha x correctness | confidence: r={r_partial:.4f}")

    # ---- Analysis 4: Domain shift alpha comparison ----
    print(f"  In-domain alpha: mean={alphas_id.mean():.4f}, std={alphas_id.std():.4f}")
    print(f"  Cross-domain alpha: mean={alphas_cd.mean():.4f}, std={alphas_cd.std():.4f}")
    ks_stat, ks_p = stats.ks_2samp(alphas_id, alphas_cd)
    reduction = (1 - alphas_cd.mean() / alphas_id.mean()) * 100
    print(f"  Alpha reduction: {reduction:.1f}%, KS p={ks_p:.2e}")

    all_results[seed] = {
        "in_domain": {"mean_alpha": float(alphas_id.mean()), "std_alpha": float(alphas_id.std()),
                      "n": int(len(alphas_id))},
        "cross_domain": {"mean_alpha": float(alphas_cd.mean()), "std_alpha": float(alphas_cd.std()),
                         "n": int(len(alphas_cd)), "accuracy": float(correct_cd.mean())},
        "split_analysis": {
            "median_alpha": float(median_alpha),
            "acc_high": float(acc_high), "acc_low": float(acc_low),
            "delta": float(acc_high - acc_low),
        },
        "point_biserial": {"r": float(r_pb), "p": float(p_pb)},
        "partial_corr": float(r_partial),
        "alpha_reduction_pct": float(reduction),
        "ks_test": {"statistic": float(ks_stat), "p": float(ks_p)},
        "per_class": class_split,
    }

# ---- Summary across seeds ----
print(f"\n{'='*60}")
print("Summary Across Seeds")
print(f"{'='*60}")
for seed, res in all_results.items():
    print(f"  {seed}: alpha_reduction={res['alpha_reduction_pct']:.1f}%, "
          f"pb_r={res['point_biserial']['r']:.4f}, partial_r={res['partial_corr']:.4f}, "
          f"split_delta={res['split_analysis']['delta']:+.4f}")

# Save
out_path = OUT_DIR / "alpha_posthoc_results.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved: {out_path}")
print("Experiment F complete!")
