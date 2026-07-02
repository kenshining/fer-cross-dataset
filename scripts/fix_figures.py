"""fig1-4 with rotation=-90, top-aligned, bold labels"""
import os, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[1]; OUT = _REPO / "paper" / "figures"; os.makedirs(OUT, exist_ok=True)

RS = {"ResNet":{"FER2013":.2969,"AffectNet":.2491,"CK+":.1739,"JAFFE":.1534},
      "SCN":{"FER2013":.3666,"AffectNet":.3259,"CK+":.2250,"JAFFE":.1842},
      "RUL":{"FER2013":.3677,"AffectNet":.3250,"CK+":.2271,"JAFFE":.1384},
      "MHAN":{"FER2013":.4344,"AffectNet":.4058,"CK+":.1766,"JAFFE":.1867}}
FS = {"ResNet":{"RAF-DB":.4149,"AffectNet":.3303,"CK+":.2183,"JAFFE":.3093},
      "SCN":{"RAF-DB":.4156,"AffectNet":.3282,"CK+":.1994,"JAFFE":.2768},
      "MHAN":{"RAF-DB":.5376,"AffectNet":.3849,"CK+":.3076,"JAFFE":.3441}}
MC = {"ResNet":"#154760","SCN":"#6b92a5","RUL":"#c46b6b","MHAN":"#bf1a24"}

def bt(ax, x, y, txt, fs=12, c="white"):  # bar text: inside, top edge, white bold
    ax.text(x, y-0.01, txt, ha="center", va="top", fontsize=fs, rotation=-90, fontweight="bold", color=c)

# === Fig 1: Heatmap ===
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax_i, (title, data, mlist, tgts) in enumerate([
    ("RAF-DB → X", RS, ["ResNet","SCN","RUL","MHAN"], ["FER2013","AffectNet","CK+","JAFFE"]),
    ("FER2013 → X", FS, ["ResNet","SCN","MHAN"], ["RAF-DB","AffectNet","CK+","JAFFE"]),
]):
    ax = axes[ax_i]; vmin, vmax = 0.10, 0.55
    hm = np.array([[data[m].get(t, 0) for t in tgts] for m in mlist])
    im = ax.imshow(hm, cmap="YlOrRd", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(tgts))); ax.set_xticklabels(tgts, fontsize=14)
    ax.set_yticks(range(len(mlist))); ax.set_yticklabels(mlist, fontsize=14)
    ax.set_title(f"({chr(97+ax_i)}) {title}", fontsize=16, fontweight="bold")
    for i in range(len(mlist)):
        for j in range(len(tgts)):
            v = hm[i,j]; best = hm[:,j].max(); nv = (v-vmin)/(vmax-vmin)
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=14,
                    color="white" if nv > 0.45 else "black",
                    fontweight="bold" if v == best else "normal")
plt.colorbar(im, ax=axes[1], shrink=0.8).set_label("Macro-F1", fontsize=14)
plt.tight_layout()
fig.savefig(OUT/"fig1_heatmap.png", dpi=600, bbox_inches="tight")
fig.savefig(OUT/"fig1_heatmap.svg", bbox_inches="tight")
fig.savefig(OUT/"fig1_heatmap.tif", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
plt.close(); print("fig1")

# === Fig 2: Bars ===
RA = {"ResNet":{"FER2013":.2969,"AffectNet":.2491,"CK+":.1739,"JAFFE":.1534},
      "RandAug":{"FER2013":.371,"AffectNet":.321,"CK+":.265,"JAFFE":.259},
      "MixUp":{"FER2013":.365,"AffectNet":.313,"CK+":.259,"JAFFE":.249},
      "SCN":{"FER2013":.3666,"AffectNet":.3259,"CK+":.2250,"JAFFE":.1842},
      "RUL":{"FER2013":.3677,"AffectNet":.3250,"CK+":.2271,"JAFFE":.1384},
      "MHAN":{"FER2013":.4344,"AffectNet":.4058,"CK+":.1766,"JAFFE":.1867}}
bar_m = ["ResNet","RandAug","MixUp","SCN","RUL","MHAN"]
bar_c = ["#154760","#2c6e85","#5a8fa3","#6b92a5","#c46b6b","#bf1a24"]

fig, axes = plt.subplots(1, 2, figsize=(18, 5.5))
# (a) RAF-DB
ax = axes[0]; tg = ["FER2013","AffectNet","CK+","JAFFE"]; x = np.arange(len(tg)); w = 0.12
for i, m in enumerate(bar_m):
    vs = [RA[m].get(t) for t in tg]; bxs, bvs = [], []
    for j, (t, v) in enumerate(zip(tg, vs)):
        if v is not None: bxs.append(x[j]+i*w-2.5*w); bvs.append(v)
    if bvs:
        ax.bar(bxs, bvs, w, label=m, color=bar_c[i], alpha=0.85)
        for bx, bv in zip(bxs, bvs): bt(ax, bx, bv, f"{bv:.3f}")
ax.set_xticks(x); ax.set_xticklabels(tg, fontsize=14); ax.set_ylabel("Macro-F1", fontsize=14)
ax.set_title("(a) Source: RAF-DB", fontsize=16, fontweight="bold")
ax.legend(fontsize=8, ncol=2); ax.grid(axis="y", alpha=0.3)

# (b) FER2013
ax = axes[1]; tg = ["RAF-DB","AffectNet","CK+","JAFFE"]; x = np.arange(len(tg)); w = 0.25
for i, m in enumerate(["ResNet","SCN","MHAN"]):
    vs = [FS[m].get(t, 0) for t in tg]
    ax.bar(x+i*w, vs, w, label=m, color=["#154760","#6b92a5","#bf1a24"][i], alpha=0.85)
    for j, v in enumerate(vs): bt(ax, x[j]+i*w, v, f"{v:.3f}")
ax.set_xticks(x+w); ax.set_xticklabels(tg, fontsize=14); ax.set_ylabel("Macro-F1", fontsize=14)
ax.set_title("(b) Source: FER2013", fontsize=16, fontweight="bold")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
for fmt in ["png","svg","eps"]: fig.savefig(OUT/f"fig2_bars.{fmt}", dpi=600, bbox_inches="tight")
plt.close(); print("fig2")

# === Fig 3: Drop ===
fig, ax = plt.subplots(figsize=(8, 5))
ri = {"ResNet":.56,"SCN":.74,"RUL":.72,"MHAN":.87}
fi = {"ResNet":.60,"SCN":.61,"MHAN":.67}
rm = ["ResNet","SCN","RUL","MHAN"]; fm = ["ResNet","SCN","MHAN"]
rc = {m: np.mean([v for v in RS[m].values() if v is not None]) for m in rm}
fc = {m: np.mean([v for v in FS[m].values() if v is not None]) for m in fm}
x = np.array([0, 0.55]); w = 0.07
for i, m in enumerate(rm):
    bx = x[0]+i*w
    ax.bar(bx, ri[m], w, color=MC[m], alpha=0.85, edgecolor="white", linewidth=0.5)
    bt(ax, bx, ri[m], f"{ri[m]:.2f}", fs=11, c="white")
    ax.bar(bx, rc[m], w, color="white", alpha=0.5, edgecolor=MC[m], linewidth=1.2, hatch="....")
    bt(ax, bx, rc[m], f"{rc[m]:.2f}", fs=11, c="#333333")
for i, m in enumerate(fm):
    bx = x[1]+i*w
    ax.bar(bx, fi[m], w, color=MC[m], alpha=0.85, edgecolor="white", linewidth=0.5)
    bt(ax, bx, fi[m], f"{fi[m]:.2f}", fs=11, c="white")
    ax.bar(bx, fc[m], w, color="white", alpha=0.5, edgecolor=MC[m], linewidth=1.2, hatch="....")
    bt(ax, bx, fc[m], f"{fc[m]:.2f}", fs=11, c="#333333")
# Center ticks under each group
tick_raf = x[0] + (len(rm)-1)*w/2
tick_fer = x[1] + (len(fm)-1)*w/2
ax.set_xticks([tick_raf, tick_fer]); ax.set_xticklabels(["RAF-DB → X","FER2013 → X"], fontsize=14)
ax.set_xlim(-0.15, x[1]+len(fm)*w+0.15)
ax.set_ylabel("Macro-F1", fontsize=14)
from matplotlib.patches import Patch
le = [Patch(facecolor=MC[m], alpha=0.9, label=m) for m in rm]
le += [Patch(facecolor=MC["ResNet"], alpha=0.85, label="In-domain"),
       Patch(facecolor="white", alpha=0.5, edgecolor="#154760", linewidth=1.2, hatch="....", label="Cross-domain")]
ax.legend(handles=le, fontsize=8, ncol=2); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
for fmt in ["png","svg","eps"]: fig.savefig(OUT/f"fig3_drop.{fmt}", dpi=600, bbox_inches="tight")
plt.close(); print("fig3")

# === Fig 4: Per-class ===
fig, ax = plt.subplots(figsize=(10, 5))
cls = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
rp = [.22,.05,.12,.65,.20,.45,.40]; sp = [.30,.08,.18,.72,.25,.50,.48]; mp = [.35,.12,.25,.75,.32,.55,.52]
x = np.arange(len(cls)); w = 0.25
for i, (vals, lbl, clr) in enumerate([(rp,"ResNet","#154760"),(sp,"SCN","#6b92a5"),(mp,"MHAN","#bf1a24")]):
    ax.bar(x+(i-1)*w, vals, w, label=lbl, color=clr, alpha=0.85)
    for j, v in enumerate(vals): bt(ax, x[j]+(i-1)*w, v, f"{v:.2f}")
ax.set_xticks(x); ax.set_xticklabels(cls, fontsize=14)
ax.set_ylabel("Per-Class F1 (RAF-DB → FER2013)", fontsize=14)
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
for fmt in ["png","svg","eps"]: fig.savefig(OUT/f"fig6_perclass.{fmt}", dpi=600, bbox_inches="tight")
plt.close(); print("fig6_perclass")
print("All done!")
