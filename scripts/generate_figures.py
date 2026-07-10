"""
Revision figure generation.
Generates 5 new figures for the revised manuscript:
  A: Full method ranking (horizontal bar, Mean +/- Std)
  B: MHAN vs ResNet per-target comparison
  C: Wavelet sensitivity (7 bases)
  D: SCN alpha stratified accuracy
  E: Multi-factor Spearman analysis

All figures follow the paper's unified color palette (Navy-Red gradient, DPI=600).
Output: PNG + SVG to 返修修改稿V1/figures/
"""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# === Configuration ===
FIG_DIR = Path("E:/scientific/小波/会议投稿/返修修改稿V1/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
RUNS = Path("E:/scientific/小波/fer-cross-dataset/runs")

# Paper's unified color palette (Navy-to-Red gradient, Nature/Science style)
NAVY = "#154760"
RED = "#bf1a24"
STEEL = "#6b92a5"
ROSE = "#c46b6b"
GRID_COLOR = "#d9e5ea"
DPI = 600
ALPHA = 0.85

# Method colors (extended from original 4-method palette)
METHOD_COLORS = {
    "SWAD": "#2ca02c",     # Green -- DG methods
    "MHAN": RED,
    "MixUp": "#8c564b",    # Brown
    "RUL": ROSE,
    "SCN": STEEL,
    "RandAug": "#e377c2",  # Pink
    "ViT-B/16": "#7f7f7f", # Gray -- transformer
    "ResNet-18": NAVY,
}

plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.unicode_minus": False,
})

# === Load all experimental data ===
with open(RUNS / "all_10seeds" / "all_results.json") as f:
    data = json.load(f)
with open(RUNS / "gpu_extra_seeds" / "all_results.json") as f:
    extra = json.load(f)
with open(RUNS / "wavelet_sensitivity" / "sensitivity_results.json") as f:
    wav_results = json.load(f)
with open(RUNS / "scn_alpha_posthoc" / "alpha_posthoc_results.json") as f:
    alpha_data = json.load(f)
with open(RUNS / "multi_factor" / "multi_factor_results_v2.json") as f:
    mf_results = json.load(f)


def get_means(prefix, paper_values, old_keys=None):
    """Merge paper values + batch1 (gpu_extra_seeds) + batch2 (all_10seeds)."""
    means = list(paper_values)
    if old_keys:
        for k in old_keys:
            if k in extra:
                means.append(extra[k]["mean"])
    for k, v in data.items():
        if k.startswith(prefix) and "mean" in v and v.get("mean", 0) > 0.01:
            means.append(v["mean"])
    return means


# ================================================================
# FIGURE A: Full Method Ranking
# Horizontal bar chart with error bars (Mean +/- Std across 10 seeds)
# ================================================================
print("FIG A: Full method ranking...")
RANK_GRADIENT = [NAVY, "#2c6e85", "#5a8fa3", "#8ab0bf", "#c99595", ROSE, RED, "#d62728"]

all_methods = {
    "SWAD": get_means("SWAD", []),
    "MHAN": get_means("MHAN", [0.301, 0.301]),
    "MixUp": get_means("MixUp", [0.297]),
    "RUL": get_means("RUL", []),
    "SCN": get_means("SCN", []),
    "RandAug": get_means("RandAug", [0.304]),
    "ViT-B/16": get_means("ViT-B", []),
    "ResNet-18": get_means("ResNet18", [0.218, 0.218]),
}

fig, ax = plt.subplots(figsize=(12, 6))
sorted_methods = sorted(all_methods.items(), key=lambda x: np.mean(x[1]), reverse=True)

for i, (name, vals) in enumerate(sorted_methods):
    mm, ms = np.mean(vals), np.std(vals, ddof=1)
    color = RANK_GRADIENT[i] if i < len(RANK_GRADIENT) else "#999999"
    ax.barh(i, mm, xerr=ms, color=color, alpha=ALPHA, capsize=3,
            edgecolor="white", linewidth=0.5)
    ax.text(mm + 0.004, i + 0.25, f"{mm:.3f}", va="center",
            fontsize=9, fontweight="bold")

ax.set_yticks(range(len(sorted_methods)))
ax.set_yticklabels([n for n, _ in sorted_methods], fontsize=10)
ax.set_xlabel("Macro-F1 (Mean +/- Std, n=10)", fontsize=11)
ax.set_xlim(0.20, 0.37)
ax.grid(axis="x", alpha=0.3, color=GRID_COLOR)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig_full_ranking.png", dpi=DPI, bbox_inches="tight")
fig.savefig(FIG_DIR / "fig_full_ranking.svg", bbox_inches="tight")
plt.close()
print("  Done.")

# ================================================================
# FIGURE B: MHAN vs ResNet Per-Target
# Grouped bar chart across 4 target datasets
# ================================================================
print("FIG B: MHAN vs ResNet per-target...")
targets = ["FER2013", "AffectNet", "CK+", "JAFFE"]
resnet_per_target = [0.297, 0.249, 0.174, 0.153]
mhan_per_target = [0.434, 0.406, 0.177, 0.187]

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(targets))
w = 0.35
ax.bar(x - w/2, resnet_per_target, w, label="ResNet-18", color=NAVY, alpha=ALPHA,
       edgecolor="white")
ax.bar(x + w/2, mhan_per_target, w, label="MHAN", color=RED, alpha=ALPHA,
       edgecolor="white")
ax.set_ylabel("Macro-F1", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(targets, fontsize=10)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3, color=GRID_COLOR)
ax.set_ylim(0, 0.55)
for i in range(4):
    delta = mhan_per_target[i] - resnet_per_target[i]
    y_pos = max(resnet_per_target[i], mhan_per_target[i]) + 0.015
    ax.annotate(f"+{delta:.3f}", (x[i], y_pos), ha="center",
                fontsize=9, fontweight="bold", color=RED)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig_mhan_vs_resnet.png", dpi=DPI, bbox_inches="tight")
fig.savefig(FIG_DIR / "fig_mhan_vs_resnet.svg", bbox_inches="tight")
plt.close()
print("  Done.")

# ================================================================
# FIGURE C: Wavelet Sensitivity
# Pearson r across 7 wavelet bases
# ================================================================
print("FIG C: Wavelet sensitivity...")
wavelets = ["Haar", "db2", "db4", "db8", "sym4", "sym8", "coif1"]
r_values = [wav_results[w]["r"] for w in wavelets]
p_values = [wav_results[w]["p"] for w in wavelets]
gradient_7 = [NAVY, "#2c6e85", "#5a8fa3", "#8ab0bf", "#c99595", ROSE, RED]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(len(wavelets)), r_values, color=gradient_7, alpha=ALPHA,
       edgecolor="white")
ax.axhline(y=0, color="black", linewidth=0.5)
ax.set_xticks(range(len(wavelets)))
ax.set_xticklabels(wavelets, fontsize=10)
ax.set_ylabel("Pearson r (n = 15)", fontsize=11)
for i, (r, p) in enumerate(zip(r_values, p_values)):
    sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
    ax.text(i, r + 0.012, f"{r:.3f}{sig}", ha="center", fontsize=9,
            fontweight="bold")
ax.grid(axis="y", alpha=0.3, color=GRID_COLOR)
ax.set_ylim(0.40, 0.78)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig_wavelet_sensitivity.png", dpi=DPI, bbox_inches="tight")
fig.savefig(FIG_DIR / "fig_wavelet_sensitivity.svg", bbox_inches="tight")
plt.close()
print("  Done.")

# ================================================================
# FIGURE D: SCN Alpha Stratified Accuracy
# Bar chart: high-alpha vs low-alpha accuracy per emotion
# ================================================================
print("FIG D: SCN alpha stratified...")
seed42 = alpha_data.get("scn_rafdb_seed42", {})
per_class = seed42.get("per_class", {})
if per_class:
    emotions = list(per_class.keys())
    acc_high = [per_class[e]["acc_high"] for e in emotions]
    acc_low = [per_class[e]["acc_low"] for e in emotions]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    plt.subplots_adjust(left=0.08, right=0.95)
    x = np.arange(len(emotions))
    w = 0.35
    bars_high = ax.bar(x - w/2, acc_high, w, label="High alpha (>= median)",
                       color=RED, alpha=ALPHA, edgecolor="white")
    bars_low = ax.bar(x + w/2, acc_low, w, label="Low alpha (< median)",
                      color=NAVY, alpha=ALPHA, edgecolor="white")

    # Value labels above each bar
    for bars in [bars_high, bars_low]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.006, f"{h:.3f}",
                        ha="center", va="bottom", fontsize=9, color="#333333",
                        fontweight="bold")

    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(emotions, fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3, color=GRID_COLOR)

    overall_delta = seed42.get("split_analysis", {}).get("delta", 0)
    ax.text(0.02, 0.95,
            f"Overall delta = {overall_delta:+.3f} (high alpha more accurate)",
            transform=ax.transAxes, fontsize=10, fontweight="bold", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_ylim(0, max(max(acc_high), max(acc_low)) * 1.25)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_scn_alpha.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_scn_alpha.svg", bbox_inches="tight")
    plt.close()
    print("  Done.")
else:
    print("  Skipped (no per-class data).")

# ================================================================
# FIGURE E: Multi-Factor Spearman Analysis
# Horizontal bar: 6 dataset features vs generalization gap
# ================================================================
print("FIG E: Multi-factor Spearman...")
features = mf_results.get("features", {})
if features:
    feat_names = list(features.keys())
    feat_labels = [
        "Resolution", "Train Samples", "Class Balance",
        "Annotators", "Environment", "Freq Distance"
    ]
    r_feat = [features[f]["spearman_r"] for f in feat_names]
    p_feat = [features[f]["spearman_p"] for f in feat_names]
    gradient_6 = [NAVY, "#2c6e85", "#5a8fa3", "#8ab0bf", "#c99595", RED]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(feat_names)), r_feat, color=gradient_6, alpha=ALPHA,
            edgecolor="white")
    ax.set_yticks(range(len(feat_names)))
    ax.set_yticklabels(feat_labels, fontsize=10)
    ax.set_xlabel("Spearman r", fontsize=11)
    ax.axvline(x=0, color="black", linewidth=0.5)
    for i, (r, p) in enumerate(zip(r_feat, p_feat)):
        sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax.text(r + 0.02, i, f"{r:+.3f}{sig}", va="center",
                fontsize=9, fontweight="bold", color=NAVY)
    ax.grid(axis="x", alpha=0.3, color=GRID_COLOR)
    ax.set_xlim(0.0, 0.95)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_multifactor.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_multifactor.svg", bbox_inches="tight")
    plt.close()
    print("  Done.")
else:
    print("  Skipped (no multi-factor data).")

print(f"\nAll 5 figures saved to: {FIG_DIR}")
