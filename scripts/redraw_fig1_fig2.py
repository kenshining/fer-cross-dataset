"""
Regenerate Fig 1 (cross-dataset heatmap) and Fig 2 (method comparison bar chart)
with the updated 8-method, 10-seed data, preserving the original visual style.
"""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

fig_dir = Path("E:/scientific/小波/会议投稿/返修修改稿V1/figures")
fig_dir.mkdir(parents=True, exist_ok=True)
base = Path("E:/scientific/小波/fer-cross-dataset/runs")

# === Data (10-seed per-target means from final analysis) ===
RAFDB = {
    "SWAD":    {"FER2013": 0.377, "AffectNet": 0.330, "CK+": 0.253, "JAFFE": 0.195},
    "MHAN":    {"FER2013": 0.391, "AffectNet": 0.341, "CK+": 0.206, "JAFFE": 0.175},
    "MixUp":   {"FER2013": 0.364, "AffectNet": 0.308, "CK+": 0.216, "JAFFE": 0.186},
    "RUL":     {"FER2013": 0.362, "AffectNet": 0.328, "CK+": 0.234, "JAFFE": 0.166},
    "SCN":     {"FER2013": 0.354, "AffectNet": 0.319, "CK+": 0.224, "JAFFE": 0.183},
    "RandAug": {"FER2013": 0.351, "AffectNet": 0.305, "CK+": 0.208, "JAFFE": 0.172},
    "ViT-B/16":{"FER2013": 0.344, "AffectNet": 0.284, "CK+": 0.195, "JAFFE": 0.231},
    "ResNet":  {"FER2013": 0.349, "AffectNet": 0.318, "CK+": 0.206, "JAFFE": 0.180},
}

METHODS = list(RAFDB.keys())
TARGETS = ["FER2013", "AffectNet", "CK+", "JAFFE"]

# Original palette extended to 8 colors
COLORS = ["#154760", "#2c6e85", "#5a8fa3", "#8ab0bf", "#c99595", "#c46b6b", "#bf1a24", "#d62728"]

plt.rcParams.update({"font.size": 10, "font.family": "serif", "axes.unicode_minus": False})
DPI = 600

# ============================================================
# FIG 1: Cross-dataset heatmap (RAF-DB source only, 8 methods)
# ============================================================
print("FIG 1: Heatmap...")
fig, ax = plt.subplots(figsize=(10, 5))
n_m, n_t = len(METHODS), len(TARGETS)
# Build heatmap transposed: targets on Y-axis, methods on X-axis
heatmap = np.zeros((n_t, n_m))
for i, t in enumerate(TARGETS):
    for j, m in enumerate(METHODS):
        heatmap[i, j] = RAFDB[m][t]

im = ax.imshow(heatmap, cmap="YlOrRd", aspect="auto", vmin=0.15, vmax=0.42)
ax.set_xticks(range(n_m))
ax.set_xticklabels(METHODS, fontsize=9, rotation=30, ha="right")
ax.set_yticks(range(n_t))
ax.set_yticklabels(TARGETS, fontsize=11)

norm = plt.Normalize(vmin=0.15, vmax=0.42)
cmap = plt.cm.YlOrRd
for i in range(n_t):
    for j in range(n_m):
        v = heatmap[i, j]
        best_col = heatmap[i, :].max()  # best in this target row
        rgba = cmap(norm(v))
        luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        color = "white" if luminance < 0.5 else "black"
        weight = "bold" if abs(v - best_col) < 0.001 else "normal"
        ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=10,
                color=color, fontweight=weight)

plt.colorbar(im, ax=ax, shrink=0.8, label="Macro-F1")
plt.tight_layout()
fig.savefig(fig_dir / "fig1_heatmap.png", dpi=DPI, bbox_inches="tight")
fig.savefig(fig_dir / "fig1_heatmap.svg", bbox_inches="tight")
plt.close()
print("  OK")

# ============================================================
# FIG 2: Method comparison bar chart (RAF-DB source only)
# ============================================================
print("FIG 2: Bar chart...")
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(TARGETS))
w = 0.10  # narrower bars for 8 methods
for i, m in enumerate(METHODS):
    vals = [RAFDB[m][t] for t in TARGETS]
    bars = ax.bar(x + i * w, vals, w, label=m, color=COLORS[i], alpha=0.85, edgecolor="white", linewidth=0.3)
    # White value label inside bar, rotated 90deg CW, centered
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., max(h - 0.010, 0.005), f"{h:.3f}",
                ha="center", va="top", fontsize=9, color="white", fontweight="bold",
                rotation=90)
ax.set_xticks(x + w * 3.5)
ax.set_xticklabels(TARGETS, fontsize=11)
ax.set_ylabel("Macro-F1", fontsize=11)
ax.legend(fontsize=8, ncol=2)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0.10, 0.45)
plt.tight_layout()
fig.savefig(fig_dir / "fig2_bars.png", dpi=DPI, bbox_inches="tight")
fig.savefig(fig_dir / "fig2_bars.svg", bbox_inches="tight")
plt.close()
print("  OK")

print(f"\nSaved to {fig_dir}")
