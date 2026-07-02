"""
Bootstrap CI for all cross-dataset comparisons.
Computes 95% confidence intervals via 10,000 resamples.
"""
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "runs" / "bootstrap_ci.json"

# Per-method per-target values from individual seeds
# RAF-DB source: 2 seeds [42, 123]
RAF = {
    "ResNet": {"FER2013": [0.297, 0.297], "AffectNet": [0.249, 0.249],
               "CK+": [0.174, 0.174], "JAFFE": [0.153, 0.153]},
    "SCN":    {"FER2013": [0.3560, 0.3772], "AffectNet": [0.3240, 0.3277],
               "CK+": [0.2173, 0.2328], "JAFFE": [0.1510, 0.2175]},
    "RUL":    {"FER2013": [0.3567, 0.3788], "AffectNet": [0.3375, 0.3125],
               "CK+": [0.2199, 0.2342], "JAFFE": [0.1916, 0.0851]},
    "MHAN":   {"FER2013": [0.4334, 0.4354], "AffectNet": [0.4045, 0.4071],
               "CK+": [0.2000, 0.1531], "JAFFE": [0.1811, 0.1923]},
}

# FER2013 source: 2 seeds [42, 456]
FER = {
    "ResNet": {"RAF-DB": [0.3728, 0.4269], "AffectNet": [0.3229, 0.3424],
               "CK+": [0.2137, 0.1765], "JAFFE": [0.3087, 0.2821]},
    "SCN":    {"RAF-DB": [0.4008, 0.4605], "AffectNet": [0.3160, 0.3308],
               "CK+": [0.1835, 0.2031], "JAFFE": [0.2156, 0.3495]},
    "MHAN":   {"RAF-DB": [0.5431, 0.5447], "AffectNet": [0.3893, 0.3760],
               "CK+": [0.3289, 0.3082], "JAFFE": [0.3750, 0.3302]},
}

def bootstrap_ci(values, n_bootstrap=10000, ci=95):
    """Compute bootstrap CI for a list of values."""
    values = np.array([v for v in values if not np.isnan(v)])
    if len(values) < 2:
        return {"mean": np.mean(values), "ci_low": np.nan, "ci_high": np.nan, "n": len(values)}
    boot_means = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.array(boot_means)
    alpha = (100 - ci) / 2
    return {"mean": np.mean(boot_means), "ci_low": np.percentile(boot_means, alpha),
            "ci_high": np.percentile(boot_means, 100-alpha), "n": len(values)}


results = {}
print("Bootstrap 95% CI for Cross-Dataset FER Benchmark\n" + "=" * 60)
for src_name, src_data in [("RAF-DB", RAF), ("FER2013", FER)]:
    results[src_name] = {}
    for method in src_data:
        results[src_name][method] = {}
        for tgt, vals in src_data[method].items():
            ci = bootstrap_ci(vals)
            results[src_name][method][tgt] = ci
            print(f"{src_name} → {tgt:10s}  {method:8s}  "
                  f"mean={ci['mean']:.4f}  CI95=[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]  n={ci['n']}")

# Overall per-method
print("\nOverall Cross-Dataset Macro-F1 (bootstrap):")
print(f"{'Method':<10} {'Mean':>8} {'CI95 low':>10} {'CI95 high':>10}")
print("-" * 40)
for method in ["ResNet", "SCN", "RUL", "MHAN"]:
    all_vals = []
    for src_data in [RAF, FER]:
        if method in src_data:
            for tgt, vals in src_data[method].items():
                all_vals.extend(vals)
    if all_vals:
        ci = bootstrap_ci(all_vals)
        print(f"{method:<10} {ci['mean']:>8.4f} {ci['ci_low']:>10.4f} {ci['ci_high']:>10.4f}")

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {OUT}")
