"""
重新生成 Figure 4：频率距离 vs 泛化下降（15 点扩展分析）。

将 AffectNet 作为源域 + 自身对纳入分析，n=15。
Panel (a): LODO 8 点。 Panel (b): 扩展 15 点（含自身对 + AffectNet 源）。

数据来源：h2_correlation_report.json（ResNet-18 跨所有源-目标对）
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

_REPO = Path(__file__).resolve().parents[1]
RUNS = _REPO / "runs"
OUT = _REPO / "paper" / "figures"

with open(RUNS / "frequency_analysis" / "h2_correlation_report.json") as f:
    report = json.load(f)

DATA = report["data_points"]

SOURCE_COLORS = {
    "RAF-DB": "#6b92a5",
    "FER2013": "#154760",
    "AffectNet": "#bf1a24",
}
SOURCE_MARKERS = {
    "RAF-DB": "o",
    "FER2013": "s",
    "AffectNet": "D",
}


def main():
    cross_only = [d for d in DATA if d["source"] != d["target"]]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # ═══════════════════════════════════════════════════════════
    # (a) 严格 LODO：8 个跨域对
    # ═══════════════════════════════════════════════════════════
    ax = axes[0]
    a_offsets = {
        "FER2013→CK+": (-20, -20),  # 左下方
    }
    lodo_pairs = [d for d in cross_only if d["source"] in ("RAF-DB", "FER2013")]
    for src in ["RAF-DB", "FER2013"]:
        pts = [d for d in lodo_pairs if d["source"] == src]
        xs = [d["composite"] for d in pts]
        ys = [d["gen_drop"] for d in pts]
        labels = [f"{d['source'][:3]}→{d['target'][:3]}" for d in pts]
        ax.scatter(xs, ys, c=SOURCE_COLORS[src], marker=SOURCE_MARKERS[src],
                   s=120, label=f"{src} source", alpha=0.85,
                   edgecolors="black", linewidth=0.5)
        for i, lab in enumerate(labels):
            key = f"{pts[i]['source']}→{pts[i]['target']}"
            ox, oy = a_offsets.get(key, (10, 10))
            ax.annotate(lab, (xs[i], ys[i]), fontsize=12, alpha=0.75,
                       textcoords="offset points", xytext=(ox, oy))

    all_lodo_xs = [d["composite"] for d in lodo_pairs]
    all_lodo_ys = [d["gen_drop"] for d in lodo_pairs]
    r8, p8 = stats.pearsonr(all_lodo_xs, all_lodo_ys)
    slope8, intercept8, _, _, _ = stats.linregress(all_lodo_xs, all_lodo_ys)
    x_line = np.linspace(0, max(all_lodo_xs) * 1.05, 50)
    ax.plot(x_line, slope8 * x_line + intercept8, "--", color="gray",
            alpha=0.6, linewidth=1.5,
            label=f"LODO only: $r$={r8:.2f}, $p$={p8:.3f}, $n$={len(lodo_pairs)}")

    ax.set_xlabel("Composite Frequency Distance (DWT-based)", fontsize=14)
    ax.set_ylabel("Generalization Drop (Δ Macro-F1)", fontsize=14)
    ax.set_title(f"(a) LODO Protocol ($n$={len(lodo_pairs)}): Frequency Distance\n"
                 "vs. Generalization Drop (ResNet-18)", fontsize=15, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right", labelspacing=0.9,
              handletextpad=0.7, borderpad=0.5)
    ax.grid(alpha=0.25)

    # ═══════════════════════════════════════════════════════════
    # (b) 扩展 15 点：含自身对和 AffectNet 源
    # ═══════════════════════════════════════════════════════════
    ax = axes[1]
    offsets = {
        "FER2013→AffectNet": (12, 10), "FER2013→CK+": (-20, -20),
        "FER2013→FER2013": (12, 8), "FER2013→JAFFE": (10, 14),
        "FER2013→RAF-DB": (6, -16),
        "RAF-DB→AffectNet": (12, -4), "RAF-DB→CK+": (-56, 10),
        "RAF-DB→FER2013": (-48, -10), "RAF-DB→JAFFE": (10, 16),
        "RAF-DB→RAF-DB": (10, 2),
        "AffectNet→AffectNet": (10, -14), "AffectNet→CK+": (12, -3),
        "AffectNet→FER2013": (12, -7), "AffectNet→JAFFE": (0, 16),
        "AffectNet→RAF-DB": (-56, 7),
    }

    for src in ["RAF-DB", "FER2013", "AffectNet"]:
        pts = [d for d in DATA if d["source"] == src]
        cross_pts = [d for d in pts if d["source"] != d["target"]]
        self_pts = [d for d in pts if d["source"] == d["target"]]

        if cross_pts:
            ax.scatter([d["composite"] for d in cross_pts],
                      [d["gen_drop"] for d in cross_pts],
                      c=SOURCE_COLORS[src], marker=SOURCE_MARKERS[src],
                      s=120, alpha=0.85,
                      edgecolors="black", linewidth=0.5)
        if self_pts:
            ax.scatter([d["composite"] for d in self_pts],
                      [d["gen_drop"] for d in self_pts],
                      c=SOURCE_COLORS[src], marker=SOURCE_MARKERS[src],
                      s=140, alpha=0.5, facecolors="none",
                      edgecolors=SOURCE_COLORS[src], linewidth=2,
                      label=f"{src} (hollow=self-pair)")

        for d in pts:
            key = f"{d['source']}→{d['target']}"
            lab = f"{d['source'][:3]}→{d['target'][:3]}"
            ox, oy = offsets.get(key, (8, 8))
            ax.annotate(lab, (d["composite"], d["gen_drop"]), fontsize=12, alpha=0.75,
                       textcoords="offset points", xytext=(ox, oy))

    all_xs = [d["composite"] for d in DATA]
    all_ys = [d["gen_drop"] for d in DATA]
    r15, p15 = stats.pearsonr(all_xs, all_ys)
    slope15, intercept15, _, _, _ = stats.linregress(all_xs, all_ys)
    x_line = np.linspace(0, max(all_xs) * 1.05, 50)
    ax.plot(x_line, slope15 * x_line + intercept15, "--", color="black",
            alpha=0.7, linewidth=2,
            label=f"All pairs: $r$={r15:.2f}, $p$={p15:.3f}, $n$={len(DATA)}")

    ax.set_xlabel("Composite Frequency Distance (DWT-based)", fontsize=14)
    ax.set_ylabel("Generalization Drop (Δ Macro-F1)", fontsize=14)
    ax.set_title(f"(b) Extended Analysis ($n$={len(DATA)}): Including\n"
                 "AffectNet Source + Self-Pairs (ResNet-18)", fontsize=15, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right", labelspacing=1.0,
              handletextpad=0.8, borderpad=0.5)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    for fmt in ["png", "svg", "eps"]:
        fig.savefig(OUT / f"fig4_freq_cross.{fmt}", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Figure 4 saved: {OUT / 'fig4_freq_cross.*'}")
    print(f"\nCorrelations:")
    print(f"  LODO (n={len(lodo_pairs)}): r={r8:.3f}, p={p8:.4f}")
    print(f"  Extended (n={len(DATA)}): r={r15:.3f}, p={p15:.4f}")
    print("Done!")


if __name__ == "__main__":
    main()
