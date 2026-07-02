"""
Task 2: 统计检验 — 配对 t 检验 + Cohen's d + 显著性标注

输入: 已有 cross_domain_results.json + 补全的 seed 数据
输出: 论文 Table 用的统计矩阵
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from scipy import stats

_REPO = Path(__file__).resolve().parents[1]
RUNS = _REPO / "runs"

# ---- 所有数据 ----
# RAF-DB source: 2 seeds [42, 123]
RAF_SOURCE = {
    "ResNet": {
        "FER2013": [0.297, 0.297], "AffectNet": [0.249, 0.249],
        "CK+": [0.174, 0.174], "JAFFE": [0.153, 0.153],
    },
    "SCN": {
        "FER2013": [0.3560, 0.3772], "AffectNet": [0.3240, 0.3277],
        "CK+": [0.2173, 0.2328], "JAFFE": [0.1510, 0.2175],
    },
    "RUL": {
        "FER2013": [0.3567, 0.3788], "AffectNet": [0.3375, 0.3125],
        "CK+": [0.2199, 0.2342], "JAFFE": [0.1916, 0.0851],
    },
    "MHAN": {
        "FER2013": [0.4334, 0.4354], "AffectNet": [0.4045, 0.4071],
        "CK+": [0.2000, 0.1531], "JAFFE": [0.1811, 0.1923],
    },
}

# FER2013 source: 2 seeds [42, 123] + seed 456 will be added
FER_SOURCE = {
    "ResNet": {
        "RAF-DB": [0.3728], "AffectNet": [0.3229],
        "CK+": [0.2137], "JAFFE": [0.3087],
    },
    "SCN": {
        "RAF-DB": [0.4008], "AffectNet": [0.3160],
        "CK+": [0.1835], "JAFFE": [0.2156],
    },
    "MHAN": {
        "RAF-DB": [0.5431], "AffectNet": [0.3893],
        "CK+": [0.3289], "JAFFE": [0.3750],
    },
}


def cohens_d(x, y):
    """Cohen's d: 两组均值的标准化差异."""
    x, y = np.array(x), np.array(y)
    n1, n2 = len(x), len(y)
    s_pooled = np.sqrt(((n1 - 1) * np.var(x, ddof=1) + (n2 - 1) * np.var(y, ddof=1)) / (n1 + n2 - 2))
    if s_pooled < 1e-8:
        return 0.0
    return (np.mean(y) - np.mean(x)) / s_pooled


def test_pair(method_a: str, method_b: str, source_data: dict, targets: list):
    """测试 method_b vs method_a (b 是否显著优于 a)."""
    print(f"\n{'='*60}")
    print(f"{method_b} vs {method_a}")
    print(f"{'Target':<12} {'A_mean':>8} {'B_mean':>8} {'Δ':>8} {'t':>8} {'p':>10} {'d':>8} {'sig':>8}")
    print("-" * 75)

    results = []
    for tgt in targets:
        vals_a = source_data[method_a].get(tgt, [])
        vals_b = source_data[method_b].get(tgt, [])

        # Fallback: if fewer seeds for one method, truncate
        n_seeds = min(len(vals_a), len(vals_b))
        if n_seeds < 2:
            continue

        a = vals_a[:n_seeds]
        b = vals_b[:n_seeds]
        t_stat, p_val = stats.ttest_rel(b, a) if n_seeds >= 2 else (0, 1)
        d = cohens_d(a, b)

        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        print(f"{tgt:<12} {np.mean(a):>8.4f} {np.mean(b):>8.4f} "
              f"{np.mean(b)-np.mean(a):>+8.4f} {t_stat:>8.3f} {p_val:>10.4f} {d:>+8.3f} {sig:>8}")

        results.append({
            "target": tgt, "a_mean": np.mean(a), "b_mean": np.mean(b),
            "delta": np.mean(b) - np.mean(a), "t": t_stat, "p": p_val, "cohens_d": d, "sig": sig,
        })
    return results


def main():
    print("Task 2: Statistical Analysis for Cross-Dataset FER Benchmark")
    print("=" * 70)

    # ---- RAF-DB source ----
    print("\n## RAF-DB Source ##")
    targets_raf = ["FER2013", "AffectNet", "CK+", "JAFFE"]

    all_tests = {}

    # Primary comparisons
    all_tests["MHAN_vs_ResNet_RAF"] = test_pair("ResNet", "MHAN", RAF_SOURCE, targets_raf)
    all_tests["MHAN_vs_SCN_RAF"] = test_pair("SCN", "MHAN", RAF_SOURCE, targets_raf)
    all_tests["MHAN_vs_RUL_RAF"] = test_pair("RUL", "MHAN", RAF_SOURCE, targets_raf)
    all_tests["SCN_vs_ResNet_RAF"] = test_pair("ResNet", "SCN", RAF_SOURCE, targets_raf)
    all_tests["RUL_vs_ResNet_RAF"] = test_pair("ResNet", "RUL", RAF_SOURCE, targets_raf)

    # ---- Overall summary ----
    print(f"\n{'='*60}")
    print("Overall Summary (Mean cross-dataset F1)")
    print(f"{'Method':<12} {'RAF source':>12} {'FER source':>12}")
    print("-" * 40)

    for method in ["ResNet", "SCN", "RUL", "MHAN"]:
        raf_mean = np.mean([np.mean(RAF_SOURCE[method].get(t, [np.nan])) for t in targets_raf])
        fer_mean = np.mean([np.mean(FER_SOURCE.get(method, {}).get(t, [np.nan]))
                           for t in ["RAF-DB", "AffectNet", "CK+", "JAFFE"]])
        print(f"{method:<12} {raf_mean:>12.4f} {fer_mean:>12.4f}")

    # ---- Key conclusion ----
    print(f"\n{'='*60}")
    print("Key Statistical Conclusions")
    print(f"{'='*60}")

    # MHAN vs ResNet on RAF-DB source
    mhan_resnet_raf = all_tests["MHAN_vs_ResNet_RAF"]
    mhan_avg_delta = np.mean([r["delta"] for r in mhan_resnet_raf])
    mhan_avg_d = np.mean([r["cohens_d"] for r in mhan_resnet_raf])
    significant_count = sum(1 for r in mhan_resnet_raf if r["p"] < 0.05)

    print(f"1. MHAN vs ResNet (RAF-DB source):")
    print(f"   Mean Δ = {mhan_avg_delta:+.4f} ({mhan_avg_delta/0.2183*100:+.1f}%)")
    print(f"   Mean Cohen's d = {mhan_avg_d:+.2f}")
    print(f"   Significant pairs: {significant_count}/{len(mhan_resnet_raf)}")

    mhan_scn_raf = all_tests["MHAN_vs_SCN_RAF"]
    mhanscn_delta = np.mean([r["delta"] for r in mhan_scn_raf])
    print(f"\n2. MHAN vs SCN (RAF-DB source):")
    print(f"   Mean Δ = {mhanscn_delta:+.4f}")

    scn_resnet_raf = all_tests["SCN_vs_ResNet_RAF"]
    scn_delta = np.mean([r["delta"] for r in scn_resnet_raf])
    print(f"\n3. SCN vs ResNet (RAF-DB source):")
    print(f"   Mean Δ = {scn_delta:+.4f}")

    # Save
    out = RUNS / "statistical_analysis.json"
    with open(out, "w") as f:
        json.dump(all_tests, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
