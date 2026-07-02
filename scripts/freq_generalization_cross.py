"""
Task 4: 频率-泛化交叉分析

将 H₁ 频率距离 + H₂ 泛化数据 + 方法数据整合到一个分析中。
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

# ---- H₁ 频率数据 ----
FREQ_STATS = {
    "CK+":     {"low": 0.9715, "high": 0.0285},
    "JAFFE":   {"low": 0.9849, "high": 0.0151},
    "FER2013": {"low": 0.9956, "high": 0.0044},
    "RAF-DB":  {"low": 0.9901, "high": 0.0099},
    "AffectNet":{"low": 0.9844, "high": 0.0156},
}

# ---- H₂ 泛化数据 ----
RAFDB_CROSS = {
    "ResNet": {"FER2013": 0.2969, "AffectNet": 0.2491, "CK+": 0.1739, "JAFFE": 0.1534},
    "SCN":    {"FER2013": 0.3666, "AffectNet": 0.3259, "CK+": 0.2250, "JAFFE": 0.1842},
    "RUL":    {"FER2013": 0.3677, "AffectNet": 0.3250, "CK+": 0.2271, "JAFFE": 0.1384},
    "MHAN":   {"FER2013": 0.4344, "AffectNet": 0.4058, "CK+": 0.1766, "JAFFE": 0.1867},
}

FER2013_CROSS = {
    "ResNet": {"RAF-DB": 0.3728, "AffectNet": 0.3229, "CK+": 0.2137, "JAFFE": 0.3087},
    "SCN":    {"RAF-DB": 0.4008, "AffectNet": 0.3160, "CK+": 0.1835, "JAFFE": 0.2156},
    "MHAN":   {"RAF-DB": 0.5431, "AffectNet": 0.3893, "CK+": 0.3289, "JAFFE": 0.3750},
}

METHOD_COLORS = {"ResNet": "#154760", "SCN": "#6b92a5", "RUL": "#c46b6b", "MHAN": "#bf1a24"}
METHOD_MARKERS = {"ResNet": "o", "SCN": "s", "RUL": "^", "MHAN": "D"}


def freq_distance(src: str, tgt: str) -> float:
    s, t = FREQ_STATS[src], FREQ_STATS[tgt]
    return np.sqrt((s["low"] - t["low"])**2 + (s["high"] - t["high"])**2)


def main():
    # ---- Build paired data ----
    data_points = []

    # RAF-DB source
    for method in ["ResNet", "SCN", "RUL", "MHAN"]:
        for tgt in ["FER2013", "AffectNet", "CK+", "JAFFE"]:
            f1 = RAFDB_CROSS[method][tgt]
            f_dist = freq_distance("RAF-DB", tgt)
            data_points.append({
                "source": "RAF-DB", "target": tgt, "method": method,
                "f1": f1, "freq_dist": f_dist,
            })

    # FER2013 source
    for method in ["ResNet", "SCN", "MHAN"]:
        for tgt in ["RAF-DB", "AffectNet", "CK+", "JAFFE"]:
            f1 = FER2013_CROSS[method][tgt]
            f_dist = freq_distance("FER2013", tgt)
            data_points.append({
                "source": "FER2013", "target": tgt, "method": method,
                "f1": f1, "freq_dist": f_dist,
            })

    # ---- Figure 6: 频率距离 vs 跨域 F1 ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # (a) Per-method scatter
    ax = axes[0]
    for method in ["ResNet", "SCN", "RUL", "MHAN"]:
        pts = [p for p in data_points if p["method"] == method]
        if not pts:
            continue
        xs = [p["freq_dist"] for p in pts]
        ys = [p["f1"] for p in pts]
        ax.scatter(xs, ys, c=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                  s=80, label=method, alpha=0.7, edgecolors="black", linewidth=0.5)
        # Regression line
        if len(xs) > 2:
            slope, intercept, r, p, _ = stats.linregress(xs, ys)
            x_line = np.linspace(min(xs), max(xs), 50)
            ax.plot(x_line, slope * x_line + intercept, "--", color=METHOD_COLORS[method],
                   alpha=0.4, linewidth=1.5)

    ax.set_xlabel("Frequency Distance (DWT-based)", fontsize=14)
    ax.set_ylabel("Cross-Dataset Macro-F1", fontsize=14)
    ax.set_title("(a) Frequency Distance vs. Generalization by Method", fontsize=16, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (b) Delta from ResNet baseline
    ax = axes[1]
    resnet_pts = {(p["source"], p["target"]): p["f1"] for p in data_points if p["method"] == "ResNet"}
    for method in ["SCN", "RUL", "MHAN"]:
        pts = [p for p in data_points if p["method"] == method]
        xs, ys, labels = [], [], []
        for p in pts:
            key = (p["source"], p["target"])
            if key in resnet_pts:
                base = resnet_pts[key]
                xs.append(p["freq_dist"])
                ys.append(p["f1"] - base)
                labels.append(f"{p['source'][:3]}→{p['target'][:3]}")
        if xs:
            ax.scatter(xs, ys, c=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                      s=80, label=method, alpha=0.7, edgecolors="black", linewidth=0.5)

    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5, linewidth=1)
    ax.set_xlabel("Frequency Distance (DWT-based)", fontsize=14)
    ax.set_ylabel("Δ Cross-Dataset F1 (vs. ResNet)", fontsize=14)
    ax.set_title("(b) Improvement over Baseline vs. Frequency Distance", fontsize=16, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    for fmt in ["png", "svg", "eps"]:
        fig.savefig(OUT / f"fig4_freq_cross.{fmt}", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Fig 6 saved: {OUT / 'fig4_freq_cross.png'}")

    # ---- Stat: correlation for each method ----
    print(f"\nFrequency-Generalization Correlations:")
    for method in ["ResNet", "SCN", "RUL", "MHAN"]:
        pts = [p for p in data_points if p["method"] == method]
        xs, ys = [p["freq_dist"] for p in pts], [p["f1"] for p in pts]
        if len(xs) > 2:
            r, p = stats.pearsonr(xs, ys)
            print(f"  {method}: r={r:+.3f}, p={p:.4f}, n={len(xs)}")

    print("Done!")


if __name__ == "__main__":
    main()
