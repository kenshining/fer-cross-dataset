"""
H₁ 验证：5个数据集的 DWT 频率签名分析

方法：
  每数据集随机抽样 300 张 → db4 3-level DWT → 频带能量占比
  统计检验：ANOVA + 两两 Kolmogorov-Smirnov
  输出：能量分布图 + 统计报告

用法：
  python fer_wavelet/scripts/frequency_signature_analysis.py
"""
from __future__ import annotations

import csv
import os
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

# ---- 配置 ----
DATA_ROOT = Path("e:/scientific/小波/data")
OUTPUT_DIR = _REPO / "runs" / "frequency_analysis"
N_SAMPLES_PER_DATASET = 300
FACE_SIZE = 224
RANDOM_SEED = 42

# ---- 数据集配置 ----
DATASET_CONFIGS = {
    "CK+": {
        "type": "image_dirs",
        "root": DATA_ROOT / "CK+" / "cohn-kanade-images",
        "label_file": DATA_ROOT / "CK+" / "Emotion_labels",
        "is_sequence": True,
    },
    "JAFFE": {
        "type": "image_dirs",
        "root": DATA_ROOT / "Jaffe",
        "emotion_dirs": ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"],
    },
    "FER2013": {
        "type": "csv_pixels",
        "csv_path": DATA_ROOT / "Fer2013" / "fer2013" / "fer2013.csv",
    },
    "RAF-DB": {
        "type": "image_dirs",
        "root": DATA_ROOT / "RAF-DB",
        "subdirs": ["Training"],
    },
    "AffectNet": {
        "type": "affectnet_csv",
        "root": DATA_ROOT / "AffectNet" / "Manually_Annotated" / "Manually_Annotated"
                  / "Manually_Annotated" / "Manually_Annotated_Images",
        "csv_path": DATA_ROOT / "AffectNet" / "Manually_Annotated_file_lists" / "training.csv",
    },
}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def _load_image_pil(path: str):
    """加载单张图像→灰度→缩放→numpy (H,W)。"""
    from PIL import Image

    img = Image.open(path).convert("L")
    img = img.resize((FACE_SIZE, FACE_SIZE), Image.BILINEAR)
    return np.array(img, dtype=np.float32)


def _dwt_energy_ratios(gray: np.ndarray):
    """
    对灰度图做 db4 3-level DWT，返回每层能量占比。

    Returns:
        levels: dict, level1/2/3 各含 LL/LH/HL/HH 能量占比
        global_low: float (0-1), LL₃ 全局能量占比
        global_high: float (0-1), 所有高频子带能量占比
    """
    import pywt

    coeffs = pywt.wavedec2(gray, "db4", level=3, mode="periodization")

    total_energy = np.sum(gray ** 2)

    result = {"level1": {}, "level2": {}, "level3": {}}

    # level 1: (cA1, (cH1, cV1, cD1))
    cA1 = coeffs[0] if len(coeffs) > 0 else None

    # Actually wavedec2 returns [cAn, (cHn, cVn, cDn), ..., (cH1, cV1, cD1)]
    cA3 = coeffs[0]
    (cH3, cV3, cD3) = coeffs[1]
    (cH2, cV2, cD2) = coeffs[2]
    (cH1, cV1, cD1) = coeffs[3]

    for name, (cH, cV, cD), (cA_name, cA) in [
        ("level3", (cH3, cV3, cD3), ("LL3", cA3)),
        ("level2", (cH2, cV2, cD2), ("LL2", None)),
        ("level1", (cH1, cV1, cD1), ("LL1", None)),
    ]:
        ll_energy = np.sum(cA ** 2) if cA is not None else 0
        lh_energy = np.sum(cH ** 2)
        hl_energy = np.sum(cV ** 2)
        hh_energy = np.sum(cD ** 2)
        level_total = ll_energy + lh_energy + hl_energy + hh_energy

        if level_total > 0:
            result[name] = {
                "LL": float(ll_energy / level_total),
                "LH": float(lh_energy / level_total),
                "HL": float(hl_energy / level_total),
                "HH": float(hh_energy / level_total),
            }
        else:
            result[name] = {"LL": 0.0, "LH": 0.0, "HL": 0.0, "HH": 0.0}

    # Global energy ratios
    # total_energy = LL₃ + sum(all detail coefficients)
    detail_energy = 0
    for cH, cV, cD in [(cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1)]:
        detail_energy += np.sum(cH ** 2) + np.sum(cV ** 2) + np.sum(cD ** 2)

    global_total = np.sum(cA3 ** 2) + detail_energy
    if global_total > 0:
        global_low = float(np.sum(cA3 ** 2) / global_total)
        global_high = float(detail_energy / global_total)
    else:
        global_low = global_high = 0.0

    return result, global_low, global_high


# ====================================================================
# 图像采样器
# ====================================================================

def sample_ckplus(config: dict, n: int) -> list[np.ndarray]:
    """CK+ 样本：每序列取最后一张（峰值表情帧）。"""
    images = []
    root = config["root"]
    label_dir = config["label_file"]

    subjects = sorted([d for d in os.listdir(root) if os.path.isdir(root / d)])
    random.shuffle(subjects)

    for subj in subjects:
        if len(images) >= n:
            break
        subj_dir = root / subj
        sessions = sorted([d for d in os.listdir(subj_dir) if os.path.isdir(subj_dir / d)])
        for sess in sessions:
            if len(images) >= n:
                break
            sess_dir = subj_dir / sess
            imgs = sorted([f for f in os.listdir(sess_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
            if imgs:
                # 取序列最后一张（峰值帧）
                last_img = sess_dir / imgs[-1]
                try:
                    gray = _load_image_pil(str(last_img))
                    images.append(gray)
                except Exception:
                    continue
    return images


def sample_jaffe(config: dict, n: int) -> list[np.ndarray]:
    """JAFFE：均匀采样各情绪类。"""
    images = []
    root = config["root"]
    emotion_dirs = config["emotion_dirs"]
    per_class = n // len(emotion_dirs) + 1

    for emo in emotion_dirs:
        emo_dir = root / emo
        if not emo_dir.is_dir():
            continue
        files = sorted([f for f in os.listdir(emo_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff"))])
        random.shuffle(files)
        for f in files[:per_class]:
            try:
                gray = _load_image_pil(str(emo_dir / f))
                images.append(gray)
            except Exception:
                continue
    random.shuffle(images)
    return images[:n]


def sample_fer2013(config: dict, n: int) -> list[np.ndarray]:
    """FER2013：从 CSV 像素数据重建图像。"""
    csv_path = config["csv_path"]
    if not csv_path.exists():
        # try alternate path
        alt = DATA_ROOT / "Fer2013" / "icml_face_data.csv"
        if alt.exists():
            csv_path = alt
        else:
            print(f"  [WARN] FER2013 CSV not found at {csv_path}")
            return []

    images = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        rows = list(reader)

    random.shuffle(rows)
    for row in rows:
        if len(images) >= n:
            break
        try:
            emotion = int(row[0])
            pixels = np.array([int(p) for p in row[1].split()], dtype=np.float32)
            img = pixels.reshape(48, 48)
            # 上采样到 224x224
            from PIL import Image
            pil_img = Image.fromarray(img.astype(np.uint8)).resize((FACE_SIZE, FACE_SIZE), Image.BILINEAR)
            images.append(np.array(pil_img, dtype=np.float32))
        except Exception:
            continue
    return images


def sample_affectnet_csv(config: dict, n: int) -> list[np.ndarray]:
    """AffectNet：直接 walk Manually_Annotated_Images 目录随机抽样。"""
    import random as _random
    images = []
    root = config["root"]
    if not root.exists():
        # fallback: search for Manually_Annotated_Images
        for dirpath, _, filenames in os.walk(DATA_ROOT / "AffectNet"):
            if dirpath.endswith("Manually_Annotated_Images"):
                root = Path(dirpath)
                break
        if not root.exists():
            print(f"  [WARN] AffectNet Manually_Annotated_Images not found")
            return images

    # Walk the manual images directory
    all_files = []
    for dirpath, _, filenames in os.walk(str(root)):
        for fn in filenames:
            if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                all_files.append(os.path.join(dirpath, fn))
        if len(all_files) >= 10000:  # stop early, enough to sample from
            break

    _random.shuffle(all_files)
    for fpath in all_files:
        if len(images) >= n:
            break
        try:
            gray = _load_image_pil(fpath)
            images.append(gray)
        except Exception:
            continue

    return images


def sample_image_dirs(config: dict, n: int) -> list[np.ndarray]:
    """通用图像目录采样器（RAF-DB, AffectNet）。"""
    images = []
    root = config["root"]

    all_files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                all_files.append(os.path.join(dirpath, fn))

    random.shuffle(all_files)
    for fpath in all_files:
        if len(images) >= n:
            break
        try:
            gray = _load_image_pil(fpath)
            images.append(gray)
        except Exception:
            continue
    return images


SAMPLERS = {
    "CK+": sample_ckplus,
    "JAFFE": sample_jaffe,
    "FER2013": sample_fer2013,
    "RAF-DB": sample_image_dirs,
    "AffectNet": sample_affectnet_csv,
}


# ====================================================================
# 主分析
# ====================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("H₁: DWT 频率签名分析")
    print(f"每个数据集抽样 {N_SAMPLES_PER_DATASET} 张")
    print(f"小波: db4, 3-level DWT")
    print("=" * 70)

    # ---- 采样 + DWT ----
    all_results = {}
    all_global_low = {}
    all_global_high = {}

    for ds_name, config in DATASET_CONFIGS.items():
        print(f"\n[{ds_name}] 采样中...")
        sampler = SAMPLERS.get(ds_name, sample_image_dirs)
        images = sampler(config, N_SAMPLES_PER_DATASET)
        print(f"  成功加载 {len(images)} 张图像")

        if len(images) == 0:
            print(f"  [SKIP] 无有效样本")
            continue

        ds_levels = {"level1": [], "level2": [], "level3": []}
        ds_global_low = []
        ds_global_high = []

        for i, gray in enumerate(images):
            levels, g_low, g_high = _dwt_energy_ratios(gray)
            for lvl_name in ds_levels:
                ds_levels[lvl_name].append(levels[lvl_name])
            ds_global_low.append(g_low)
            ds_global_high.append(g_high)

            if (i + 1) % 100 == 0:
                print(f"  处理中... {i + 1}/{len(images)}")

        all_results[ds_name] = {
            "levels": ds_levels,
            "n_samples": len(images),
        }
        all_global_low[ds_name] = ds_global_low
        all_global_high[ds_name] = ds_global_high

        # 打印统计
        print(f"  {ds_name}: LL₃全局能量比 = {np.mean(ds_global_low):.4f} ± {np.std(ds_global_low):.4f}")
        print(f"           高频全局能量比 = {np.mean(ds_global_high):.4f} ± {np.std(ds_global_high):.4f}")

    # ---- 可视化 ----
    ds_names = list(all_results.keys())
    if len(ds_names) < 2:
        print("数据集数量不足，无法分析")
        return

    # Unified red-navy palette consistent with other figures
    # Navy (wild datasets) → Red (lab datasets)
    ds_colors = {
        "AffectNet": "#154760",
        "RAF-DB":    "#2c6e85",
        "FER2013":   "#6b92a5",
        "CK+":       "#c99595",
        "JAFFE":     "#bf1a24",
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("DWT Frequency Signature Analysis (db4, 3-level)\n"
                 f"N={N_SAMPLES_PER_DATASET} per dataset",
                 fontsize=14, fontweight="bold")

    # (a) LL₃ 全局能量比：箱线图
    ax = axes[0, 0]
    data = [all_global_low[d] for d in ds_names]
    bp = ax.boxplot(data, labels=ds_names, patch_artist=True)
    for patch, d in zip(bp["boxes"], ds_names):
        patch.set_facecolor(ds_colors[d])
        patch.set_alpha(0.6)
    ax.set_ylabel("LL₃ Global Energy Ratio")
    ax.set_title("(a) Low-Frequency Energy (higher = smoother dataset)")
    ax.grid(axis="y", alpha=0.3)

    # (b) 全局高频能量比：箱线图
    ax = axes[0, 1]
    data = [all_global_high[d] for d in ds_names]
    bp = ax.boxplot(data, labels=ds_names, patch_artist=True)
    for patch, d in zip(bp["boxes"], ds_names):
        patch.set_facecolor(ds_colors[d])
        patch.set_alpha(0.6)
    ax.set_ylabel("High-Frequency Global Energy Ratio")
    ax.set_title("(b) High-Frequency Energy (higher = more detail/noise)")
    ax.grid(axis="y", alpha=0.3)

    # (c) Level 3 子带分解
    ax = axes[0, 2]
    x = np.arange(len(ds_names))
    width = 0.2
    band_colors = ["#154760", "#6b92a5", "#c99595", "#bf1a24"]  # LL→HH: navy→red
    for j, band in enumerate(["LL", "LH", "HL", "HH"]):
        means = [np.mean([s[band] for s in all_results[d]["levels"]["level3"]]) for d in ds_names]
        ax.bar(x + j * width, means, width, label=band, alpha=0.8, color=band_colors[j])
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(ds_names, fontsize=8)
    ax.set_ylabel("Energy Ratio at Level 3")
    ax.set_title("(c) Level-3 Subband Energy Distribution")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # (d) Level 2 子带分解
    ax = axes[1, 0]
    for j, band in enumerate(["LL", "LH", "HL", "HH"]):
        means = [np.mean([s[band] for s in all_results[d]["levels"]["level2"]]) for d in ds_names]
        ax.bar(x + j * width, means, width, label=band, alpha=0.8, color=band_colors[j])
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(ds_names, fontsize=8)
    ax.set_ylabel("Energy Ratio at Level 2")
    ax.set_title("(d) Level-2 Subband Energy Distribution")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # (e) Level 1 子带分解
    ax = axes[1, 1]
    for j, band in enumerate(["LL", "LH", "HL", "HH"]):
        means = [np.mean([s[band] for s in all_results[d]["levels"]["level1"]]) for d in ds_names]
        ax.bar(x + j * width, means, width, label=band, alpha=0.8, color=band_colors[j])
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(ds_names, fontsize=8)
    ax.set_ylabel("Energy Ratio at Level 1")
    ax.set_title("(e) Level-1 Subband Energy Distribution")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # (f) 散点图：LL vs HH（每个样本一个点）
    ax = axes[1, 2]
    for d in ds_names:
        low_vals = all_global_low[d]
        high_vals = all_global_high[d]
        # 采样到相同数量以保持可读性
        n_plot = min(200, len(low_vals))
        idx = np.random.choice(len(low_vals), n_plot, replace=False)
        ax.scatter(
            [low_vals[i] for i in idx],
            [high_vals[i] for i in idx],
            c=ds_colors[d], label=d, alpha=0.5, s=8,
        )
    ax.set_xlabel("LL₃ Global Energy Ratio")
    ax.set_ylabel("High-Freq Global Energy Ratio")
    ax.set_title("(f) Sample Distribution: Smooth vs. Detailed")
    ax.legend(fontsize=7, markerscale=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    # Save to figures directory with consistent naming
    figures_dir = _REPO / "runs" / "figures"
    os.makedirs(figures_dir, exist_ok=True)
    for fmt in ["png", "svg", "eps"]:
        fig_path = figures_dir / f"fig4_freq_signature.{fmt}"
        fig.savefig(fig_path, dpi=600, bbox_inches="tight")
    # Also save to original output directory
    fig_path_orig = OUTPUT_DIR / "frequency_signature.png"
    fig.savefig(fig_path_orig, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"\n图表已保存: {figures_dir / 'fig4_freq_signature.*'}")

    # ---- 统计检验 ----
    print(f"\n{'=' * 70}")
    print("统计检验")
    print("=" * 70)

    from scipy import stats

    # (1) ANOVA on global_low
    groups_low = [all_global_low[d] for d in ds_names]
    f_stat, p_anova = stats.f_oneway(*groups_low)
    print(f"\n[ANOVA] LL₃ 全局能量比跨数据集差异:")
    print(f"  F = {f_stat:.4f}, p = {p_anova:.2e}")
    print(f"  判定: {'✅ 显著差异 (p < 0.01)' if p_anova < 0.01 else '⚠️ 不显著 (p ≥ 0.01)'}")

    groups_high = [all_global_high[d] for d in ds_names]
    f_stat_h, p_anova_h = stats.f_oneway(*groups_high)
    print(f"\n[ANOVA] 高频全局能量比跨数据集差异:")
    print(f"  F = {f_stat_h:.4f}, p = {p_anova_h:.2e}")
    print(f"  判定: {'✅ 显著差异 (p < 0.01)' if p_anova_h < 0.01 else '⚠️ 不显著 (p ≥ 0.01)'}")

    # (2) 两两 KS 检验
    print(f"\n[KS检验] 两两数据集 LL₃ 能量分布差异:")
    print(f"{'Dataset A':<12} {'Dataset B':<12} {'KS stat':<10} {'p-value':<12} {'显著?':<10}")
    print("-" * 56)
    sig_count_ll = 0
    for i, a in enumerate(ds_names):
        for j, b in enumerate(ds_names):
            if i >= j:
                continue
            ks_stat, p_ks = stats.ks_2samp(all_global_low[a], all_global_low[b])
            sig = "✅" if p_ks < 0.01 else ("⚠️" if p_ks < 0.05 else "❌")
            if p_ks < 0.01:
                sig_count_ll += 1
            print(f"{a:<12} {b:<12} {ks_stat:<10.4f} {p_ks:<12.2e} {sig:<10}")

    print(f"\n  显著差异对数 (p<0.01): {sig_count_ll}/{len(ds_names) * (len(ds_names) - 1) // 2}")

    # (3) H₁ 最终判定
    total_pairs = len(ds_names) * (len(ds_names) - 1) // 2
    print(f"\n{'=' * 70}")
    print("H₁ 最终判定")
    print("=" * 70)

    h1_passed = p_anova < 0.01 and sig_count_ll >= total_pairs * 0.5
    if h1_passed:
        print(f"✅ H₁ 成立: 不同数据集具有系统性的频率特征差异")
        print(f"   ANOVA p={p_anova:.2e} < 0.01")
        print(f"   显著 KS 对数: {sig_count_ll}/{total_pairs} ≥ 50%")
    else:
        print(f"⚠️ H₁ 不满足: ANOVA p={p_anova:.2e}, 显著KS对 {sig_count_ll}/{total_pairs}")

    # 保存数据
    import json
    report = {
        "h1_passed": h1_passed,
        "anova_low": {"F": float(f_stat), "p": float(p_anova)},
        "anova_high": {"F": float(f_stat_h), "p": float(p_anova_h)},
        "n_significant_pairs": sig_count_ll,
        "n_total_pairs": total_pairs,
        "datasets": {},
    }
    for d in ds_names:
        report["datasets"][d] = {
            "n_samples": all_results[d]["n_samples"],
            "global_low_mean": float(np.mean(all_global_low[d])),
            "global_low_std": float(np.std(all_global_low[d])),
            "global_high_mean": float(np.mean(all_global_high[d])),
            "global_high_std": float(np.std(all_global_high[d])),
        }

    report_path = OUTPUT_DIR / "frequency_analysis_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n报告已保存: {report_path}")

    # 打印数据集间频率距离矩阵
    print(f"\n[频率距离矩阵] 基于 LL₃ 能量比的 Wasserstein 距离:")
    print(f"{'':<12}", end="")
    for d in ds_names:
        print(f"{d:<12}", end="")
    print()
    for a in ds_names:
        print(f"{a:<12}", end="")
        for b in ds_names:
            w_dist = stats.wasserstein_distance(all_global_low[a], all_global_low[b])
            print(f"{w_dist:<12.4f}", end="")
        print()

    return h1_passed


if __name__ == "__main__":
    main()
