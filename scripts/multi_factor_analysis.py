"""
Multi-factor dataset characteristic analysis.
Computes Spearman correlations between 6 dataset-level features and cross-dataset
generalization gap using composite frequency distance with 15 source-target pairs.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
from scipy import stats

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

OUT_DIR = _REPO / "runs" / "multi_factor"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Dataset features ----
DATASETS = {
    "CK+":       {"res":640*490, "n_train":0,     "balance":0.82, "annotators":1,  "env":0, "E_LL":0.9715, "E_HF":0.0285},
    "JAFFE":     {"res":256*256, "n_train":0,     "balance":0.95, "annotators":1,  "env":0, "E_LL":0.9849, "E_HF":0.0151},
    "FER2013":   {"res":48*48,   "n_train":28709, "balance":0.89, "annotators":1,  "env":1, "E_LL":0.9956, "E_HF":0.0044},
    "RAF-DB":    {"res":100*100, "n_train":8332,  "balance":0.87, "annotators":40, "env":1, "E_LL":0.9901, "E_HF":0.0099},
    "AffectNet": {"res":100*100, "n_train":283901,"balance":0.83, "annotators":1,  "env":1, "E_LL":0.9844, "E_HF":0.0156},
}

# ---- Features to analyze ----
FEATURES = ["res", "n_train", "balance", "annotators", "env", "freq_dist"]
FEATURE_LABELS = {
    "res": "Resolution (px)",
    "n_train": "Training Samples",
    "balance": "Class Balance",
    "annotators": "Num. Annotators",
    "env": "Environment (lab/wild)",
    "freq_dist": "Freq. Distance (composite)",
}

# ---- 15 source-target pairs (matching paper's expanded analysis) ----
# Each entry: (src, tgt, generalization_gap)
# gap = 1 - cross_F1 (self-pairs: gap=0)
PAIRS = [
    # Cross-dataset (12 pairs)
    ("RAF-DB", "FER2013",   0.7031),
    ("RAF-DB", "AffectNet", 0.7509),
    ("RAF-DB", "CK+",       0.8261),
    ("RAF-DB", "JAFFE",     0.8466),
    ("FER2013", "RAF-DB",   0.6272),
    ("FER2013", "AffectNet",0.6771),
    ("FER2013", "CK+",      0.7863),
    ("FER2013", "JAFFE",    0.6913),
    ("AffectNet", "FER2013",0.6373),
    ("AffectNet", "RAF-DB", 0.5476),
    ("AffectNet", "CK+",    0.8054),
    ("AffectNet", "JAFFE",  0.7642),
    # Self-pairs (3 pairs, gap=0 by construction)
    ("RAF-DB", "RAF-DB",    0.0),
    ("FER2013", "FER2013",  0.0),
    ("AffectNet", "AffectNet",0.0),
]

def composite_freq_dist(src, tgt):
    """Euclidean distance between (E_LL, E_HF) vectors."""
    s, t = DATASETS[src], DATASETS[tgt]
    return np.sqrt((s["E_LL"]-t["E_LL"])**2 + (s["E_HF"]-t["E_HF"])**2)

def feature_diff(src, tgt, feat):
    """Compute feature difference for a source-target pair."""
    s, t = DATASETS[src], DATASETS[tgt]
    if feat == "freq_dist":
        return composite_freq_dist(src, tgt)
    if feat == "env":
        return abs(s["env"] - t["env"])  # binary diff
    # Normalize by max value across datasets
    max_val = max(abs(DATASETS[d][feat]) for d in DATASETS) + 1e-8
    return abs(s[feat] - t[feat]) / max_val

print("="*60)
print("Experiment G v2: Multi-Factor Analysis (fixed)")
print(f"Pairs: {len(PAIRS)} (12 cross + 3 self)")
print("="*60)

# Build feature difference matrix and gap vector
n_pairs = len(PAIRS)
X = np.zeros((n_pairs, len(FEATURES)))
gaps = np.zeros(n_pairs)

for k, (src, tgt, gap) in enumerate(PAIRS):
    for j, feat in enumerate(FEATURES):
        X[k, j] = feature_diff(src, tgt, feat)
    gaps[k] = gap

# ---- Spearman correlation per feature ----
print(f"\n[Spearman] Feature difference vs generalization gap (n={n_pairs}):")
corr_results = {}
for j, feat in enumerate(FEATURES):
    r, p = stats.spearmanr(X[:, j], gaps)
    corr_results[feat] = {"spearman_r": round(float(r),4), "spearman_p": float(p)}
    sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
    print(f"  {FEATURE_LABELS[feat]:<35} r={r:+.4f}, p={p:.4f} {sig}")

# ---- Rank features ----
ranked = sorted(corr_results.items(), key=lambda x: abs(x[1]["spearman_r"]), reverse=True)
print(f"\nFeature ranking (by |Spearman r|):")
for i, (feat, res) in enumerate(ranked, 1):
    print(f"  {i}. {FEATURE_LABELS[feat]:<35} |r|={abs(res['spearman_r']):.4f}")

# ---- Pearson (for comparison with paper's r=0.70) ----
print(f"\n[Pearson] freq_dist vs gap (for paper comparison):")
for j, feat in enumerate(FEATURES):
    r_p, p_p = stats.pearsonr(X[:, j], gaps)
    if feat == "freq_dist":
        print(f"  Composite freq distance: r={r_p:.4f}, p={p_p:.4f}")
        print(f"  (Paper reports: r=0.70, p=0.004 with 15 pairs -- should be close)")

# ---- Inter-feature correlation ----
print(f"\n[Inter-feature Spearman correlation]:")
inter_corr = np.zeros((len(FEATURES), len(FEATURES)))
for j1 in range(len(FEATURES)):
    for j2 in range(len(FEATURES)):
        r, _ = stats.spearmanr(X[:, j1], X[:, j2])
        inter_corr[j1, j2] = r

# Print top collinearities
collinear = []
for j1 in range(len(FEATURES)):
    for j2 in range(j1+1, len(FEATURES)):
        if abs(inter_corr[j1, j2]) > 0.5:
            collinear.append((FEATURES[j1], FEATURES[j2], inter_corr[j1,j2]))
collinear.sort(key=lambda x: abs(x[2]), reverse=True)
for f1, f2, r in collinear[:5]:
    print(f"  {FEATURE_LABELS[f1]} <-> {FEATURE_LABELS[f2]}: r={r:+.4f}")

# ---- Save ----
output = {
    "description": "Multi-factor analysis with composite freq distance + 15 pairs (v2)",
    "n_pairs": n_pairs,
    "features": {feat: corr_results[feat] for feat in FEATURES},
    "ranked": [(feat, corr_results[feat]) for feat, _ in ranked],
    "inter_feature_corr": {
        FEATURES[j1]: {FEATURES[j2]: round(float(inter_corr[j1,j2]),4) for j2 in range(len(FEATURES))}
        for j1 in range(len(FEATURES))
    },
}

out_path = OUT_DIR / "multi_factor_results_v2.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {out_path}")

# ---- Paper discussion summary ----
print(f"\n{'='*60}")
print("Key Findings for Paper Discussion")
print(f"{'='*60}")
top3 = ranked[:3]
for i, (feat, res) in enumerate(top3, 1):
    direction = "positive" if res["spearman_r"] > 0 else "negative"
    print(f"  {i}. {FEATURE_LABELS[feat]}: r={res['spearman_r']:+.4f} ({direction})")
print("\n  -> Frequency distance is {position} among {n} features.".format(
    position=["1st","2nd","3rd","4th","5th","6th"][[f for f,_ in ranked].index("freq_dist")],
    n=len(FEATURES)))
print("Experiment G v2 complete!")
