"""
H₂ 验证：频率距离 vs. 跨数据集泛化下降的相关性分析

方法：
  1. 用已训练的 RGB 模型做跨数据集评估
  2. 计算源-目标数据集间的频率距离（基于 H₁）
  3. 皮尔逊相关系数 → 判定 H₂

用法：
  python fer_wavelet/scripts/h2_correlation_analysis.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.models import FERWaveletModel
from src.dataset_registry import REGISTRY, create_loaders
from src.train import cross_domain_evaluate

# ---- 配置 ----
RUNS_ROOT = _REPO / "runs"
FREQ_REPORT = RUNS_ROOT / "frequency_analysis" / "frequency_analysis_report.json"
OUTPUT_DIR = RUNS_ROOT / "frequency_analysis"
PROJECT_ROOT = _REPO.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "datasets.yaml"
BATCH_SIZE = 8
FACE_SIZE = 224
NUM_CLASSES = 7
SEED = 42


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_config():
    """加载配置，支持环境变量覆盖数据根目录。"""
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    data_root_env = os.environ.get("FER_DATA_ROOT")
    if data_root_env:
        config["root"] = data_root_env
    if not config.get("root"):
        config["root"] = str(PROJECT_ROOT / "data")
    return config


def load_frequency_data():
    """从 H₁ 分析报告加载频率数据。"""
    with open(FREQ_REPORT, encoding="utf-8") as f:
        report = json.load(f)
    freq_stats = {}
    for ds_name, info in report["datasets"].items():
        freq_stats[ds_name] = {
            "low_mean": info["global_low_mean"],
            "low_std": info["global_low_std"],
            "high_mean": info["global_high_mean"],
            "high_std": info["global_high_std"],
        }
    return freq_stats


def compute_frequency_distance(stat_a: dict, stat_b: dict) -> dict:
    """计算两个数据集之间的频率距离（多度量）。"""
    low_diff = abs(stat_a["low_mean"] - stat_b["low_mean"])
    high_diff = abs(stat_a["high_mean"] - stat_b["high_mean"])
    euclidean = np.sqrt(low_diff**2 + high_diff**2)
    var_diff = abs(stat_a["low_std"] - stat_b["low_std"])
    composite = np.sqrt(low_diff**2 + high_diff**2 + var_diff**2)
    return {
        "low_diff": low_diff,
        "high_diff": high_diff,
        "euclidean": euclidean,
        "var_diff": var_diff,
        "composite": composite,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = get_device()
    print(f"设备: {device}")
    print("=" * 70)
    print("H2: 频率距离 vs. 跨数据集泛化下降")
    print("=" * 70)

    # ---- 加载 H1 数据 ----
    freq_stats = load_frequency_data()
    all_freq_datasets = sorted(freq_stats.keys())
    print(f"已有频率数据的源数据集: {all_freq_datasets}")

    # ---- 加载配置 ----
    config = load_config()

    # ---- 构建测试 DataLoader ----
    print("\n[1] 构建测试 DataLoaders...")
    test_loaders = {}

    # 注册表名称映射
    ds_name_map = {
        "CK+": "ckplus",
        "JAFFE": "jaffe",
        "FER2013": "fer2013",
        "RAF-DB": "rafdb",
        "AffectNet": "affectnet",
    }

    for display_name in all_freq_datasets:
        reg_name = ds_name_map.get(display_name, display_name.lower())
        if reg_name not in REGISTRY:
            print(f"  [SKIP] {display_name}: 未在注册表中找到")
            continue
        try:
            _, val_loader = create_loaders(
                dataset_name=reg_name, config=config,
                batch_size=BATCH_SIZE, num_workers=0,
                seed=SEED, smoke_samples=None,
                device_type="cpu",
            )
            if val_loader is None:
                # ckplus/jaffe: create_loaders returns (loader, None), reuse loader for eval
                tl, _ = create_loaders(
                    dataset_name=reg_name, config=config,
                    batch_size=BATCH_SIZE, num_workers=0,
                    seed=SEED, smoke_samples=None,
                    device_type="cpu",
                )
                val_loader = tl

            if val_loader is not None:
                test_loaders[display_name] = val_loader
                n_samples = len(val_loader.dataset)
                print(f"  {display_name}: {n_samples} 样本")
            else:
                print(f"  [SKIP] {display_name}: 无法创建 DataLoader")
        except Exception as e:
            print(f"  [ERROR] {display_name}: {e}")
            import traceback
            traceback.print_exc()

    # ---- 可用的训练好的 RGB 模型 ----
    source_models = {
        "FER2013": RUNS_ROOT / "fer2013_rgb" / "best.pt",
        "RAF-DB": RUNS_ROOT / "rafdb_rgb" / "best.pt",
        "AffectNet": RUNS_ROOT / "affectnet_rgb" / "best.pt",
    }

    # ---- 跨数据集评估 ----
    print("\n[2] 跨数据集评估...")
    results = {}

    for src_name, model_path in source_models.items():
        if not model_path.exists():
            print(f"  [SKIP] {src_name}: 模型文件不存在 ({model_path})")
            continue
        if src_name not in test_loaders:
            print(f"  [SKIP] {src_name}: 无测试 loader")
            continue

        for tgt_name, tgt_loader in test_loaders.items():
            print(f"  {src_name} -> {tgt_name} ...", end=" ", flush=True)
            try:
                eval_result = cross_domain_evaluate(
                    model_path=model_path,
                    target_loader=tgt_loader,
                    mode="rgb",
                    face_size=FACE_SIZE,
                    yolo_weights=None,
                    device=device,
                    num_classes=NUM_CLASSES,
                )
                f1 = eval_result["macro_f1"]
                acc = eval_result["acc"]
                results[(src_name, tgt_name)] = {
                    "macro_f1": f1,
                    "acc": acc,
                }
                print(f"macro_f1={f1:.4f}, acc={acc:.4f}")
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()

    if len(results) < 3:
        print(f"结果不足 ({len(results)} 个数据点)，跳过相关性分析")
        return

    # ---- 获取域内性能 ----
    print("\n[3] 获取域内性能作为参考...")
    in_domain_f1 = {}
    for src_name in source_models:
        reg_name = ds_name_map.get(src_name, src_name.lower())
        run_meta_path = RUNS_ROOT / f"{reg_name}_rgb" / "run_meta.json"
        if run_meta_path.exists():
            with open(run_meta_path) as f:
                meta = json.load(f)
            in_domain_f1[src_name] = meta.get("best_val_macro_f1", None)
            print(f"  {src_name}: 域内 macro_f1 = {in_domain_f1[src_name]:.4f}")
        else:
            # Fallback: use same-dataset cross-domain result
            sd = results.get((src_name, src_name), {})
            if sd:
                in_domain_f1[src_name] = sd["macro_f1"]
            else:
                in_domain_f1[src_name] = None

    # ---- 计算泛化下降 + 频率距离 ----
    print("\n[4] 计算泛化下降 vs. 频率距离...")
    data_points = []

    for (src, tgt), perf in results.items():
        if perf["macro_f1"] <= 0:
            continue
        if src not in freq_stats or tgt not in freq_stats:
            continue
        if in_domain_f1.get(src) is None:
            continue

        in_d = in_domain_f1[src]
        gen_drop = max(0, in_d - perf["macro_f1"])
        gen_ratio = perf["macro_f1"] / max(in_d, 1e-8)

        fdist = compute_frequency_distance(freq_stats[src], freq_stats[tgt])
        dp = {
            "source": src,
            "target": tgt,
            "in_domain_f1": in_d,
            "cross_domain_f1": perf["macro_f1"],
            "gen_drop": gen_drop,
            "gen_ratio": gen_ratio,
            **fdist,
        }
        data_points.append(dp)
        print(f"  {src}->{tgt}: 域内={in_d:.4f}, 跨域={perf['macro_f1']:.4f}, "
              f"下降={gen_drop:.4f}, 频率距离={fdist['composite']:.6f}")

    if len(data_points) < 3:
        print(f"数据点不足 ({len(data_points)})，无法做相关性分析")
        return

    # ---- 相关性分析 ----
    from scipy import stats

    print(f"\n[5] 皮尔逊相关性分析 ({len(data_points)} 个数据点):")

    gen_drops = [dp["gen_drop"] for dp in data_points]
    gen_ratios = [dp["gen_ratio"] for dp in data_points]

    dist_metrics = ["low_diff", "high_diff", "euclidean", "var_diff", "composite"]
    correlations = {}

    for metric in dist_metrics:
        dists = [dp[metric] for dp in data_points]
        r_drop, p_drop = stats.pearsonr(dists, gen_drops)
        r_ratio, p_ratio = stats.pearsonr(dists, gen_ratios)
        correlations[metric] = {
            "r_gen_drop": r_drop, "p_gen_drop": p_drop,
            "r_gen_ratio": r_ratio, "p_gen_ratio": p_ratio,
        }
        sig_drop = "✅" if p_drop < 0.05 else "⚠️"
        sig_ratio = "✅" if p_ratio < 0.05 else "⚠️"
        print(f"  {metric:>12}: r_drop={r_drop:+.4f} {sig_drop} (p={p_drop:.4f}), "
              f"r_ratio={r_ratio:+.4f} {sig_ratio} (p={p_ratio:.4f})")

    # ---- H2 判定 ----
    best_metric = max(correlations, key=lambda m: abs(correlations[m]["r_gen_drop"]))
    best_r = correlations[best_metric]["r_gen_drop"]
    best_p = correlations[best_metric]["p_gen_drop"]

    h2_passed = any(
        abs(c["r_gen_drop"]) > 0.5 and c["p_gen_drop"] < 0.05
        for c in correlations.values()
    )

    print(f"\n{'=' * 70}")
    print("H2 最终判定")
    print("=" * 70)
    if h2_passed:
        print(f"✅ H2 成立: 频率距离与跨数据集泛化下降存在显著相关")
        print(f"   最优度量: {best_metric}, r={best_r:.4f}, p={best_p:.4f}")
    else:
        print(f"⚠️ H2 不满足: |r| ≤ 0.5 或 p ≥ 0.05")
        if abs(best_r) > 0.3:
            print(f"   有中等趋势: {best_metric}, r={best_r:.4f}, p={best_p:.4f}")
        else:
            print(f"   最优度量: {best_metric}, r={best_r:.4f}, p={best_p:.4f}")

    # ---- 可视化 ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("H2: Frequency Distance vs. Cross-Dataset Generalization",
                 fontsize=14, fontweight="bold")

    # (a) gen_drop vs composite
    ax = axes[0]
    dists = [dp["composite"] for dp in data_points]
    ax.scatter(dists, gen_drops, c="#2196F3", s=80, zorder=5)
    for dp in data_points:
        ax.annotate(f"{dp['source'][:3]}->{dp['target'][:3]}",
                    (dp["composite"], dp["gen_drop"]), fontsize=6, alpha=0.7)
    if len(dists) > 2:
        slope, intercept, r_val, p_val, _ = stats.linregress(dists, gen_drops)
        xs = np.linspace(min(dists), max(dists), 100)
        ax.plot(xs, slope * xs + intercept, "--", color="red", alpha=0.6)
        ax.text(0.05, 0.95, f"r={r_val:.3f}, p={p_val:.3f}",
                transform=ax.transAxes, fontsize=10, va="top")
    ax.set_xlabel("Frequency Distance (Composite)")
    ax.set_ylabel("Generalization Drop (in-domain - cross-domain F1)")
    ax.set_title("(a) Gen Drop vs Freq Distance")
    ax.grid(alpha=0.3)

    # (b) gen_ratio vs low_diff
    ax = axes[1]
    dists_low = [dp["low_diff"] for dp in data_points]
    ax.scatter(dists_low, gen_ratios, c="#4CAF50", s=80, zorder=5)
    for dp in data_points:
        ax.annotate(f"{dp['source'][:3]}->{dp['target'][:3]}",
                    (dp["low_diff"], dp["gen_ratio"]), fontsize=6, alpha=0.7)
    if len(dists_low) > 2:
        slope, intercept, r_val, p_val, _ = stats.linregress(dists_low, gen_ratios)
        xs = np.linspace(min(dists_low), max(dists_low), 100)
        ax.plot(xs, slope * xs + intercept, "--", color="red", alpha=0.6)
        ax.text(0.05, 0.95, f"r={r_val:.3f}, p={p_val:.3f}",
                transform=ax.transAxes, fontsize=10, va="top")
    ax.set_xlabel("LL3 Energy Ratio Difference")
    ax.set_ylabel("Generalization Ratio (cross/in-domain F1)")
    ax.set_title("(b) Gen Ratio vs Low-Freq Distance")
    ax.grid(alpha=0.3)

    # (c) 热力图
    ax = axes[2]
    all_ds = sorted(set(dp["source"] for dp in data_points) | set(dp["target"] for dp in data_points))
    n_ds = len(all_ds)
    heatmap = np.full((n_ds, n_ds), np.nan)
    for dp in data_points:
        if dp["source"] in all_ds and dp["target"] in all_ds:
            i, j = all_ds.index(dp["source"]), all_ds.index(dp["target"])
            heatmap[i, j] = dp["cross_domain_f1"]
    im = ax.imshow(heatmap, cmap="YlOrRd", aspect="auto", vmin=0)
    ax.set_xticks(range(n_ds))
    ax.set_xticklabels(all_ds, fontsize=8, rotation=45)
    ax.set_yticks(range(n_ds))
    ax.set_yticklabels(all_ds, fontsize=8)
    ax.set_title("(c) Cross-Dataset F1 Matrix")
    for i in range(n_ds):
        for j in range(n_ds):
            if not np.isnan(heatmap[i, j]):
                color = "white" if heatmap[i, j] < 0.4 else "black"
                ax.text(j, i, f"{heatmap[i, j]:.3f}", ha="center", va="center",
                        fontsize=7, fontweight="bold", color=color)
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "h2_correlation.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n图表已保存: {fig_path}")

    # 保存报告
    report = {
        "h2_passed": h2_passed,
        "n_data_points": len(data_points),
        "best_metric": best_metric,
        "best_r": best_r,
        "best_p": best_p,
        "correlations": correlations,
        "data_points": data_points,
    }
    report_path = OUTPUT_DIR / "h2_correlation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"报告已保存: {report_path}")


if __name__ == "__main__":
    main()
