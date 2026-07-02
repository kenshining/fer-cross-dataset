"""
统一实验入口：支持 5 个数据集 × 4 组消融 + 跨域评估 + 断点续训。

用法:
    # 1) 在 RAF-DB 上运行全部四组消融
    python fer_wavelet/scripts/run_experiments.py --dataset rafdb

    # 2) 全数据集全模式运行（断点续训）
    python fer_wavelet/scripts/run_experiments.py --dataset all --resume

    # 3) 全域消融 + 跨域评估
    python fer_wavelet/scripts/run_experiments.py --dataset all --cross-domain ckplus,jaffe --resume

环境变量:
    FER_DATA_ROOT: 数据根目录（如 E:/scientific/小波/data）
    SMOKE_EPOCHS:   冒烟测试轮数（如 1）
    SMOKE_SAMPLES:  冒烟测试样本数（如 100）
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.dataset_registry import REGISTRY, compute_class_weights, create_loaders
from src.models import AblationMode
from src.train import (
    CHECKPOINT_FILE,
    cross_domain_evaluate,
    setup_gpu,
    train_one_run,
)


def _ts() -> str:
    return time.strftime("%H:%M:%S", time.localtime())


def _project_root() -> Path:
    return _REPO.parent


# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="双分支小波-CNN 面部表情识别 — 统一实验入口",
    )
    p.add_argument("--dataset", default="rafdb",
                    help="数据集名称 (fer2013, rafdb, affectnet, ckplus, jaffe, all)，默认 rafdb")
    p.add_argument("--modes", default=None,
                    help="消融模式，逗号分隔 (rgb,low_only,high_only,fusion)，默认全量四组")
    p.add_argument("--cross-domain", default=None,
                    help="跨域评估目标数据集 (逗号分隔, 如 ckplus,jaffe)")
    p.add_argument("--epochs", type=int, default=None, help="覆盖配置中的 epochs")
    p.add_argument("--batch-size", type=int, default=None, help="覆盖配置中的 batch_size")
    p.add_argument("--lr", type=float, default=None, help="覆盖配置中的 lr")
    p.add_argument("--seed", type=int, default=None, help="覆盖配置中的 seed")
    p.add_argument("--memory-fraction", type=float, default=0.5,
                    help="GPU 显存上限比例 (默认 0.5，即 8GB x 50pct = 4GB)")
    p.add_argument("--resume", action="store_true",
                    help="断点续训模式：跳过已完成 mode，自动恢复未完成的 mode")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 工具：判断 mode 是否已完成 / 可续训
# ---------------------------------------------------------------------------
def _mode_is_done(out_dir: Path) -> bool:
    """如果存在已完成的 run_meta（含 finished_unix），视为已完成。"""
    meta = out_dir / "run_meta.json"
    if not meta.is_file():
        return False
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
        return "finished_unix" in data
    except Exception:
        return False


def _mode_can_resume(out_dir: Path) -> bool:
    """存在 checkpoint.pt 表示可续训。"""
    return (out_dir / CHECKPOINT_FILE).is_file()


# ---------------------------------------------------------------------------
# 主要实验流程
# ---------------------------------------------------------------------------
def run_experiment(
    dataset_name: str,
    config: dict,
    modes: list[AblationMode],
    args: argparse.Namespace,
    runs_root: Path,
) -> dict:
    """对指定数据集运行所有消融模式，返回 summary。"""
    train_cfg = config["training"]
    pre_cfg = config["preprocess"]

    face_size = int(pre_cfg["face_size"])
    yolo_weights = (
        os.environ.get("YOLO_WEIGHTS") or pre_cfg.get("yolo_weights") or ""
    ).strip() or None
    batch_size = args.batch_size or int(train_cfg["batch_size"])
    epochs = args.epochs or int(os.environ.get("SMOKE_EPOCHS", train_cfg["epochs"]))
    lr = args.lr or float(train_cfg["lr"])
    seed = args.seed or int(train_cfg["seed"])
    num_workers = int(train_cfg.get("num_workers", 0))
    pretrained = os.environ.get("FER_PRETRAINED", "1").strip() != "0"
    smoke_samples = int(os.environ.get("SMOKE_SAMPLES", "0")) or None
    num_classes = int(config["datasets"].get(dataset_name, {}).get("num_classes", 7))

    print(f"数据集: {dataset_name} | 分类数: {num_classes}", flush=True)
    print(f"超参数: batch={batch_size}, epochs={epochs}, lr={lr}, seed={seed}", flush=True)
    print(f"早停: patience=5, monitor=val_macro_f1, 最佳权重自动保存", flush=True)
    print(f"LR调度: ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-6)", flush=True)

    # 先创建 DataLoader（不初始化 CUDA，节省页面文件）
    train_loader, val_loader = create_loaders(
        dataset_name=dataset_name, config=config,
        batch_size=batch_size, num_workers=0,
        seed=seed, smoke_samples=smoke_samples,
        device_type="cpu",
    )

    # DataLoader 创建完成后再初始化 GPU
    device = setup_gpu(memory_fraction=args.memory_fraction)
    if train_loader is None or val_loader is None:
        print(f"⚠  {dataset_name} 未生成完整的训练/验证 DataLoader，跳过", flush=True)
        return {}

    print(f"训练样本: {len(train_loader.dataset)} | "
          f"验证样本: {len(val_loader.dataset)}", flush=True)

    class_weights = compute_class_weights(
        dataset_name, config, train_loader, num_classes=num_classes
    )
    print(f"类别权重: {class_weights.tolist()}", flush=True)

    # 逐模式消融（支持断点续训）
    summary: dict[str, object] = {"dataset": dataset_name, "modes": {}}
    num_modes = len(modes)

    for mode_idx, mode in enumerate(modes):
        out_dir = runs_root / f"{dataset_name}_{mode}"

        # 续训判断
        if args.resume:
            if _mode_is_done(out_dir):
                print(f"\n  ⏭  {mode} 已完成，跳过", flush=True)
                # 读取已完成的结果
                meta = json.loads((out_dir / "run_meta.json").read_text(encoding="utf-8"))
                summary["modes"][mode] = {
                    "best_val_macro_f1": meta.get("best_val_macro_f1", 0),
                    "run_dir": str(out_dir),
                }
                continue
            if _mode_can_resume(out_dir):
                print(f"\n  [{_ts()}] 🔄 {mode} 发现 checkpoint，断点续训", flush=True)

        print(f"\n{'='*60}", flush=True)
        print(f"[{_ts()}] 模式 [{mode_idx+1}/{num_modes}]: {mode} → 输出: {out_dir}", flush=True)
        print(f"{'='*60}", flush=True)

        # 每轮模式前清理缓存，释放 GPU 显存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 看门狗重试：页面文件不足时进程可能被杀，自动恢复
        MAX_RETRIES = 3
        res = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                res = train_one_run(
                    train_loader=train_loader, val_loader=val_loader,
                    mode=mode, num_classes=num_classes,
                    epochs=epochs, lr=lr, seed=seed,
                    face_size=face_size, yolo_weights=yolo_weights,
                    pretrained=pretrained, out_dir=out_dir,
                    device=device, class_weights=class_weights,
                    single_epoch=bool(args.resume),
                )
                break  # 成功则跳出重试
            except (SystemExit, Exception) as e:
                if attempt < MAX_RETRIES:
                    print(f"  ⚠ {mode} 训练中断 ({e})，"
                          f"第 {attempt}/{MAX_RETRIES-1} 次重试...", flush=True)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    time.sleep(5)
                else:
                    print(f"  ✗ {mode} 训练在 {MAX_RETRIES} 次重试后仍失败，跳过", flush=True)
                    raise

        if res is None:
            continue
        summary["modes"][mode] = {
            "best_val_macro_f1": res["best_val_macro_f1"],
            "run_dir": res["run_dir"],
        }

        # 单 epoch 模式：检查是否还有未完成的 epoch
        if args.resume:
            meta_path = out_dir / "run_meta.json"
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if "finished_unix" not in meta:
                    print(f"  [单epoch] {mode} 尚未完成所有 epoch，"
                          f"退出进程以释放资源，看门狗将自动重启。", flush=True)
                    summary["_exit_early"] = True
                    break  # 跳出 mode 循环，触发脚本退出

    return summary


def run_cross_domain(
    source_dataset: str,
    target_datasets: list[str],
    config: dict,
    args: argparse.Namespace,
    runs_root: Path,
    src_summary: dict,
) -> dict:
    """跨域评估：用 source 训练的最佳融合模型 -> 在 target 上 zero-shot 测试。"""
    train_cfg = config["training"]
    pre_cfg = config["preprocess"]
    face_size = int(pre_cfg["face_size"])
    yolo_weights = (
        os.environ.get("YOLO_WEIGHTS") or pre_cfg.get("yolo_weights") or ""
    ).strip() or None
    batch_size = args.batch_size or int(train_cfg["batch_size"])
    num_workers = int(train_cfg.get("num_workers", 0))
    num_classes = 7
    device = setup_gpu(memory_fraction=args.memory_fraction)

    results: dict[str, object] = {}

    for target_name in target_datasets:
        if target_name not in REGISTRY:
            print(f"⚠ 未知目标数据集: {target_name}", flush=True)
            continue

        print(f"\n{'='*60}", flush=True)
        print(f"跨域: {source_dataset} → {target_name}", flush=True)
        print(f"{'='*60}", flush=True)

        target_loader, _ = create_loaders(
            dataset_name=target_name, config=config,
            batch_size=batch_size, num_workers=num_workers,
            seed=int(train_cfg["seed"]), device_type=device.type,
        )
        if target_loader is None:
            print(f"⚠ 无法加载目标数据集 {target_name}", flush=True)
            continue

        print(f"目标域样本数: {len(target_loader.dataset)}", flush=True)

        mode_results = {}
        for mode in src_summary.get("modes", {}):
            mode_info = src_summary["modes"][mode]
            ckpt_path = Path(mode_info["run_dir"]) / "best.pt"
            if not ckpt_path.is_file():
                print(f"⚠ 未找到 checkpoint: {ckpt_path}", flush=True)
                continue

            eval_res = cross_domain_evaluate(
                model_path=ckpt_path, target_loader=target_loader,
                mode=mode, face_size=face_size,
                yolo_weights=yolo_weights, device=device,
                num_classes=num_classes,
            )
            mode_results[mode] = {
                "acc": eval_res["acc"],
                "macro_f1": eval_res["macro_f1"],
                "per_class_f1": eval_res["per_class_f1"],
            }
            print(f"  mode={mode}: acc={eval_res['acc']:.4f}, "
                  f"macro_f1={eval_res['macro_f1']:.4f}", flush=True)

        results[target_name] = mode_results

    return results


def main() -> None:
    args = parse_args()
    root = _project_root()
    cfg_path = root / "configs" / "datasets.yaml"
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    data_root_env = os.environ.get("FER_DATA_ROOT")
    if data_root_env:
        config["root"] = data_root_env
    if not config.get("root"):
        config["root"] = str(root / "data")

    runs_root = _REPO / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    # 消融模式
    if args.modes:
        modes = [m.strip() for m in args.modes.split(",")]
        for m in modes:
            if m not in ("rgb", "low_only", "high_only", "fusion"):
                raise SystemExit(f"无效消融模式: {m}")
    else:
        modes = ["rgb", "low_only", "high_only", "fusion"]

    # 数据集列表
    datasets = ["fer2013", "rafdb", "affectnet"] if args.dataset == "all" else [args.dataset]

    # 执行实验
    all_summaries: dict[str, object] = {}
    for ds_name in datasets:
        if ds_name not in REGISTRY:
            print(f"⚠ 跳过未知数据集: {ds_name}", flush=True)
            continue
        info = REGISTRY[ds_name]
        if info.get("cross_domain_only"):
            print(f"⚠ {ds_name} 仅用于跨域评估，跳过独立训练", flush=True)
            continue

        print(f"\n{'#'*60}", flush=True)
        print(f"[{_ts()}] # 开始实验: {ds_name}", flush=True)
        print(f"{'#'*60}", flush=True)

        summary = run_experiment(ds_name, config, modes, args, runs_root)
        all_summaries[ds_name] = summary

        # 单 epoch 提前退出：释放资源后让看门狗重启
        if isinstance(summary, dict) and summary.pop("_exit_early", False):
            print(f"  [单epoch] 数据集 {ds_name} 尚未完成，"
                  f"清理后退出。", flush=True)
            (runs_root / f"{ds_name}_ablation_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            # 写入部分摘要后退出进程
            summary_path = runs_root / "all_experiments_summary.json"
            summary_path.write_text(
                json.dumps(all_summaries, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"部分摘要已写入: {summary_path}", flush=True)
            print("单 epoch 完成，退出以释放系统资源。看门狗将自动重启。", flush=True)
            raise SystemExit(0)

        (runs_root / f"{ds_name}_ablation_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # 跨域评估
    if args.cross_domain:
        target_list = [t.strip() for t in args.cross_domain.split(",")]
        for ds_name in datasets:
            if ds_name not in all_summaries:
                continue
            src_sum = all_summaries[ds_name]
            if not isinstance(src_sum, dict):
                continue

            cd_results = run_cross_domain(
                ds_name, target_list, config, args, runs_root, src_sum
            )
            cd_path = runs_root / f"{ds_name}_cross_domain.json"
            cd_path.write_text(
                json.dumps(cd_results, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"跨域结果已写入: {cd_path}", flush=True)
            all_summaries[f"{ds_name}_cross_domain"] = cd_results

    summary_path = runs_root / "all_experiments_summary.json"
    summary_path.write_text(
        json.dumps(all_summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n全局摘要已写入: {summary_path}", flush=True)
    print("完成！", flush=True)


if __name__ == "__main__":
    main()
