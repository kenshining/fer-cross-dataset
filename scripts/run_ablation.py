"""
四组消融一键跑：rgb / low_only / high_only / fusion（FER2013）。

用法（在仓库根目录 `小波` 下）::
    set FER_DATA_ROOT=E:\\scientific\\小波\\data
    set SMOKE_EPOCHS=1
    python fer_wavelet/scripts/run_ablation.py

或在 `fer_wavelet` 目录::
    python scripts/run_ablation.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import yaml

# 保证可导入 src.*
_REPO = Path(__file__).resolve().parents[1]  # fer_wavelet
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from torch.utils.data import DataLoader, Subset

from src.dataset_fer2013 import FER2013Dataset, fer2013_collate_fn
from src.models import AblationMode
from src.train import compute_class_weights_from_subset, random_train_val_indices, train_one_run


def _project_root() -> Path:
    """小波 根目录（configs 所在）。"""
    return _REPO.parent


def main() -> None:
    root = _project_root()
    cfg_path = root / "configs" / "datasets.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    fer_cfg = cfg["datasets"]["fer2013"]
    if not fer_cfg.get("enabled", False):
        raise SystemExit("datasets.fer2013.enabled 为 false，请在 datasets.yaml 中启用。")

    data_root = Path(os.environ.get("FER_DATA_ROOT") or cfg.get("root") or "")
    csv_rel = Path(fer_cfg["path"]) / fer_cfg["train_csv"]
    csv_path = (data_root / csv_rel).resolve() if str(data_root) else (root / "data" / csv_rel).resolve()
    if not csv_path.is_file():
        raise SystemExit(
            f"未找到 FER2013 CSV: {csv_path}\n"
            "请设置环境变量 FER_DATA_ROOT 指向含 fer2013/train.csv 的数据根目录，"
            "或在 configs/datasets.yaml 中填写 root。"
        )

    train_cfg = cfg["training"]
    pre_cfg = cfg["preprocess"]
    face_size = int(pre_cfg["face_size"])
    yolo_weights = (os.environ.get("YOLO_WEIGHTS") or pre_cfg.get("yolo_weights") or "").strip() or None
    num_classes = int(fer_cfg["num_classes"])
    batch_size = int(train_cfg["batch_size"])
    epochs = int(os.environ.get("SMOKE_EPOCHS", train_cfg["epochs"]))
    lr = float(train_cfg["lr"])
    seed = int(train_cfg["seed"])
    num_workers = int(train_cfg.get("num_workers", 0))
    pretrained = os.environ.get("FER_PRETRAINED", "1").strip() != "0"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_ds = FER2013Dataset(csv_path)
    n = len(full_ds)
    train_idx, val_idx = random_train_val_indices(n, val_fraction=0.1, seed=seed)
    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)
    class_w = compute_class_weights_from_subset(full_ds, train_idx, num_classes=num_classes)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=fer2013_collate_fn,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=fer2013_collate_fn,
        pin_memory=device.type == "cuda",
    )

    modes: list[AblationMode] = ["rgb", "low_only", "high_only", "fusion"]
    runs_root = _REPO / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {}

    for mode in modes:
        out_dir = runs_root / f"fer2013_{mode}"
        print(f"=== mode={mode} -> {out_dir} ===", flush=True)
        res = train_one_run(
            train_loader=train_loader,
            val_loader=val_loader,
            mode=mode,
            num_classes=num_classes,
            epochs=epochs,
            lr=lr,
            seed=seed,
            face_size=face_size,
            yolo_weights=yolo_weights,
            pretrained=pretrained,
            out_dir=out_dir,
            device=device,
            class_weights=class_w,
        )
        summary[mode] = {
            "best_val_macro_f1": res["best_val_macro_f1"],
            "run_dir": res["run_dir"],
        }

    (runs_root / "ablation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("已写入", runs_root / "ablation_summary.json", flush=True)


if __name__ == "__main__":
    main()
