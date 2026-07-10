"""
Wavelet sensitivity analysis for the FER cross-dataset benchmark.
Computes frequency-generalization correlation (Pearson r) across 7 wavelet bases
to verify robustness of the DWT-based diagnostic framework.
"""
from __future__ import annotations
import csv, json, os, random, sys, time
from pathlib import Path
import numpy as np
from scipy import stats

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

DATA_ROOT = Path("e:/scientific/小波/data")
OUT_DIR = _REPO / "runs" / "wavelet_sensitivity"
os.makedirs(OUT_DIR, exist_ok=True)

N_SAMPLES = 300
FACE_SIZE = 224
SEED = 42

random.seed(SEED); np.random.seed(SEED)

WAVELETS = [("Haar","haar"),("db2","db2"),("db4","db4"),("db8","db8"),
            ("sym4","sym4"),("sym8","sym8"),("coif1","coif1")]

# ---- Image loading ----
def _load_gray(path):
    from PIL import Image
    img = Image.open(path).convert("L")
    img = img.resize((FACE_SIZE, FACE_SIZE), Image.BILINEAR)
    return np.array(img, dtype=np.float32)

# ---- DWT energy ratios ----
def _dwt_energy(gray, wavelet="db4"):
    import pywt
    coeffs = pywt.wavedec2(gray, wavelet, level=3, mode="periodization")
    cA3, (cH3,cV3,cD3) = coeffs[0], coeffs[1]
    (cH2,cV2,cD2), (cH1,cV1,cD1) = coeffs[2], coeffs[3]
    detail_e = sum(np.sum(c**2) for c in [cH3,cV3,cD3,cH2,cV2,cD2,cH1,cV1,cD1])
    total = np.sum(cA3**2) + detail_e
    if total > 0:
        return float(np.sum(cA3**2)/total), float(detail_e/total)
    return 0.0, 0.0

# ---- Samplers ----
def sample_ckplus(root, n):
    imgs = []
    for subj in sorted(d for d in os.listdir(root) if os.path.isdir(root/d)):
        if len(imgs) >= n: break
        subj_dir = root/subj
        for sess in sorted(d for d in os.listdir(subj_dir) if os.path.isdir(subj_dir/d)):
            if len(imgs) >= n: break
            sess_dir = subj_dir/sess
            files = sorted(f for f in os.listdir(sess_dir) if f.lower().endswith((".png",".jpg",".jpeg")))
            if files:
                try: imgs.append(_load_gray(str(sess_dir/files[-1])))
                except: continue
    return imgs

def sample_jaffe(root, n):
    imgs = []
    emotions = ["anger","disgust","fear","happiness","neutral","sadness","surprise"]
    per = n//len(emotions)+1
    for emo in emotions:
        emo_dir = root/emo
        if not emo_dir.is_dir(): continue
        files = sorted(f for f in os.listdir(emo_dir) if f.lower().endswith((".png",".jpg",".jpeg",".tiff")))
        random.shuffle(files)
        for f in files[:per]:
            try: imgs.append(_load_gray(str(emo_dir/f)))
            except: continue
    random.shuffle(imgs)
    return imgs[:n]

def sample_fer2013(csv_path, n):
    if not csv_path.exists():
        alt = DATA_ROOT/"Fer2013"/"icml_face_data.csv"
        csv_path = alt if alt.exists() else csv_path
    imgs = []
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.reader(f))[1:]
    random.shuffle(rows)
    for row in rows:
        if len(imgs) >= n: break
        try:
            pixels = np.array([int(p) for p in row[1].split()], dtype=np.float32)
            img = pixels.reshape(48,48)
            from PIL import Image
            pil = Image.fromarray(img.astype(np.uint8)).resize((FACE_SIZE,FACE_SIZE), Image.BILINEAR)
            imgs.append(np.array(pil, dtype=np.float32))
        except: continue
    return imgs

def sample_dir(root, n):
    imgs = []
    all_files = []
    for dp,_,fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith((".png",".jpg",".jpeg")): all_files.append(os.path.join(dp,fn))
        if len(all_files) >= 10000: break
    random.shuffle(all_files)
    for fp in all_files:
        if len(imgs) >= n: break
        try: imgs.append(_load_gray(fp))
        except: continue
    return imgs

def sample_affectnet(root, n):
    manual_root = None
    for dp,_,_ in os.walk(DATA_ROOT/"AffectNet"):
        if dp.endswith("Manually_Annotated_Images"):
            manual_root = Path(dp); break
    if manual_root is None:
        print("  [WARN] AffectNet manual images not found")
        return []
    return sample_dir(manual_root, n)

# ---- Dataset configs ----
DATASETS = {
    "CK+":       lambda: sample_ckplus(DATA_ROOT/"CK+"/"cohn-kanade-images", N_SAMPLES),
    "JAFFE":     lambda: sample_jaffe(DATA_ROOT/"Jaffe", N_SAMPLES),
    "FER2013":   lambda: sample_fer2013(DATA_ROOT/"Fer2013"/"fer2013"/"fer2013.csv", N_SAMPLES),
    "RAF-DB":    lambda: sample_dir(DATA_ROOT/"RAF-DB", N_SAMPLES),
    "AffectNet": lambda: sample_affectnet(DATA_ROOT/"AffectNet", N_SAMPLES),
}

# ---- Generalization data (ResNet-18 15-pair) ----
GEN_DATA = {
    ("RAF-DB","FER2013"):0.2969, ("RAF-DB","AffectNet"):0.2491,
    ("RAF-DB","CK+"):0.1739, ("RAF-DB","JAFFE"):0.1534,
    ("FER2013","RAF-DB"):0.3728, ("FER2013","AffectNet"):0.3229,
    ("FER2013","CK+"):0.2137, ("FER2013","JAFFE"):0.3087,
    ("AffectNet","FER2013"):0.3627, ("AffectNet","RAF-DB"):0.4524,
    ("AffectNet","CK+"):0.1946, ("AffectNet","JAFFE"):0.2358,
    ("RAF-DB","RAF-DB"):1.0, ("FER2013","FER2013"):1.0, ("AffectNet","AffectNet"):1.0,
}

print("="*60)
print("Experiment E: Wavelet Sensitivity Analysis")
print("="*60)

# Preload all images (cache per dataset)
print("\n[Preload] Sampling images...")
all_images = {}
for ds_name, sampler in DATASETS.items():
    imgs = sampler()
    all_images[ds_name] = imgs
    print(f"  {ds_name}: {len(imgs)} images")

results = {}

for wav_label, wav_name in WAVELETS:
    t0 = time.time()
    print(f"\n--- {wav_label} ({wav_name}) ---")

    # Compute E_LL / E_HF per dataset
    ds_stats = {}
    for ds_name, images in all_images.items():
        ell_vals, ehf_vals = [], []
        for gray in images:
            ell, ehf = _dwt_energy(gray, wav_name)
            ell_vals.append(ell); ehf_vals.append(ehf)
        ds_stats[ds_name] = {
            "E_LL": float(np.mean(ell_vals)),
            "E_HF": float(np.mean(ehf_vals)),
            "n": len(images),
        }

    # Compute frequency distance + Pearson r
    freq_dists, gen_gaps = [], []
    for (src, tgt), gen_f1 in GEN_DATA.items():
        if src not in ds_stats or tgt not in ds_stats: continue
        s, t = ds_stats[src], ds_stats[tgt]
        d = np.sqrt((s["E_LL"]-t["E_LL"])**2 + (s["E_HF"]-t["E_HF"])**2)
        freq_dists.append(d)
        gen_gaps.append(1.0 - gen_f1)

    r, p = stats.pearsonr(freq_dists, gen_gaps)
    elapsed = time.time() - t0

    results[wav_label] = {
        "wavelet": wav_name, "n_pairs": len(freq_dists),
        "r": round(float(r),4), "p": float(p),
        "significant": bool(p < 0.05),
        "E_LL": {k: round(v["E_LL"],4) for k,v in ds_stats.items()},
        "elapsed_s": round(elapsed,1),
    }
    sig_mark = "SIG" if p < 0.05 else "ns"
    print(f"  n={len(freq_dists)}, r={r:.4f}, p={p:.4f}  [{sig_mark}]  ({elapsed:.0f}s)")

# ---- Save ----
out_path = OUT_DIR / "sensitivity_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_path}")

# ---- Summary table ----
print(f"\n{'Wavelet':<10} {'r':>8} {'p':>8} {'Sig':>6} {'n':>6}")
print("-"*40)
for wl, _ in WAVELETS:
    r = results.get(wl,{})
    if "error" in r: print(f"{wl:<10} ERROR: {r['error'][:30]}")
    else: print(f"{wl:<10} {r['r']:>8.4f} {r['p']:>8.4f} {'YES' if r['significant'] else 'NO':>6} {r['n_pairs']:>6}")

# R range
r_vals = [results[w]["r"] for w,_ in WAVELETS if "r" in results.get(w,{})]
print(f"\nR range: [{min(r_vals):.4f}, {max(r_vals):.4f}], mean={np.mean(r_vals):.4f}, std={np.std(r_vals):.4f}")
print("Experiment E complete!")
