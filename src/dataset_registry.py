"""
数据集工厂注册表：将数据集名称映射到对应的 Dataset 类、collate_fn 和默认配置。

所有数据集的标签统一映射（0-6）:
  Index | Emotion  | FER2013 | RAF-DB      | AffectNet | CK+ | JAFFE
  -------|----------|---------|-------------|-----------|-----|-------
  0     | angry    | 0       | Anger       | 6         | 1   | anger
  1     | disgust  | 1       | Disgust     | 5         | 3   | disgust
  2     | fear     | 2       | Fear        | 4         | 4   | fear
  3     | happy    | 3       | Happiness   | 1         | 5   | happiness
  4     | sad      | 4       | Sadness     | 2         | 6   | sadness
  5     | surprise | 5       | Surprise    | 3         | 7   | surprise
  6     | neutral  | 6       | Neutral     | 0         | 0   | neutral
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.dataset_fer2013 import FER2013Dataset, fer2013_collate_fn
from src.dataset_rafdb import RAFDBDataset, rafdb_collate_fn
from src.dataset_affectnet import AffectNetDataset, affectnet_collate_fn
from src.dataset_ckplus import CKPlusDataset, ckplus_collate_fn
from src.dataset_jaffe import JAFFEDataset, jaffe_collate_fn
from src.train import random_train_val_indices

# ---------------------------------------------------------------------------
# 数据集信息注册表
# ---------------------------------------------------------------------------
DatasetInfo = dict[str, Any]

REGISTRY: dict[str, DatasetInfo] = {
    "fer2013": {
        "dataset_cls": FER2013Dataset,
        "collate_fn": fer2013_collate_fn,
        "csv_relpath": "Fer2013/train.csv",
        "use_csv": True,
        "has_official_val": False,
        "val_fraction": 0.1,
    },
    "rafdb": {
        "dataset_cls": RAFDBDataset,
        "collate_fn": rafdb_collate_fn,
        "relpath": "RAF-DB",
        "has_official_val": True,
        "train_split": "train",
        "val_split": "test",
    },
    "affectnet": {
        "dataset_cls": AffectNetDataset,
        "collate_fn": affectnet_collate_fn,
        "relpath": "AffectNet",
        "has_official_val": True,
        "train_csv": "training.csv",
        "val_csv": "validation.csv",
    },
    "ckplus": {
        "dataset_cls": CKPlusDataset,
        "collate_fn": ckplus_collate_fn,
        "relpath": "CK+",
        "has_official_val": False,
        "val_fraction": 0.0,
        "cross_domain_only": True,
    },
    "jaffe": {
        "dataset_cls": JAFFEDataset,
        "collate_fn": jaffe_collate_fn,
        "relpath": "Jaffe",
        "has_official_val": False,
        "val_fraction": 0.0,
        "cross_domain_only": True,
    },
}


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def _resolve_data_root(config: dict, dataset_name: str) -> Path:
    dscfg = config.get("datasets", {}).get(dataset_name, {})
    rel = dscfg.get("path") or REGISTRY[dataset_name].get("relpath", "")
    data_root = Path(config.get("root") or "data") / rel
    return data_root


def create_loaders(
    dataset_name: str,
    config: dict,
    batch_size: int,
    num_workers: int,
    seed: int,
    smoke_samples: int | None = None,
    device_type: str = "cpu",
) -> tuple[DataLoader, DataLoader | None]:
    """创建训练/验证 DataLoader。

    参数:
        dataset_name: 数据集名称（fer2013 / rafdb / affectnet）
        config: datasets.yaml 配置字典
        batch_size: 批大小
        num_workers: DataLoader 工作进程数
        seed: 随机种子（用于内部划分验证集）
        smoke_samples: 限制样本数用于冒烟测试
        device_type: 设备类型

    返回:
        (train_loader, val_loader)  — 验证集可为 None（仅跨域数据集）
    """
    info = REGISTRY[dataset_name]
    cls = info["dataset_cls"]
    collate_fn = info["collate_fn"]
    ds_cfg = config.get("datasets", {}).get(dataset_name, {})

    data_root = _resolve_data_root(config, dataset_name)

    if dataset_name == "fer2013":
        csv_rel = info["csv_relpath"]
        csv_path = data_root.parent / csv_rel
        full_ds = cls(csv_path)
    elif dataset_name == "rafdb":
        full_ds = cls(data_root, split="train")
    elif dataset_name == "affectnet":
        max_samples = smoke_samples or ds_cfg.get("max_train_samples")
        train_ds = cls(data_root, split="train", max_samples=max_samples)
        val_ds = cls(data_root, split="val", max_samples=smoke_samples)
        return _build_loaders(train_ds, val_ds, collate_fn, batch_size, num_workers, device_type)
    elif dataset_name in ("ckplus", "jaffe"):
        full_ds = cls(data_root)
        loader = DataLoader(
            full_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=collate_fn,
            pin_memory=device_type == "cuda" and num_workers == 0,
        )
        return loader, None
    else:
        raise ValueError(f"未知数据集: {dataset_name}")

    # FER2013 / RAF-DB：内部随机划分
    n = len(full_ds)
    val_fraction = info.get("val_fraction", 0.1)
    train_idx, val_idx = random_train_val_indices(n, val_fraction=val_fraction, seed=seed)
    if smoke_samples is not None:
        train_idx = train_idx[: min(smoke_samples, len(train_idx))]
        val_idx = val_idx[: max(1, min(smoke_samples // 4, len(val_idx)))]

    train_subset = Subset(full_ds, train_idx)
    val_subset = Subset(full_ds, val_idx)

    return _build_loaders(train_subset, val_subset, collate_fn, batch_size, num_workers, device_type)


def _build_loaders(
    train_ds: Dataset, val_ds: Dataset, collate_fn,
    batch_size: int, num_workers: int, device_type: str,
) -> tuple[DataLoader, DataLoader]:
    # Windows 下多 worker + pin_memory 易触发页面文件错误 1455
    pin = device_type == "cuda" and num_workers == 0
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=pin,
    )
    return train_loader, val_loader


def _extract_labels(ds: Dataset, indices: list[int] | None) -> list[int]:
    """从数据集提取标签列表（不加载图片）。"""
    for attr in ("samples", "rows"):
        try:
            items = getattr(ds, attr)
            if indices is not None:
                return [items[i][1] for i in indices]
            return [s[1] for s in items]
        except (AttributeError, IndexError, TypeError):
            continue
    # 回退：遍历 DataLoader
    all_labels: list[int] = []
    loader = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=0)
    for _, ys in loader:
        all_labels.extend(ys.tolist())
    return all_labels


def compute_class_weights(
    dataset_name: str,
    config: dict,
    train_loader: DataLoader,
    num_classes: int = 7,
) -> torch.Tensor:
    """从训练集统计类别频率并计算加权 CE 权重（不加载图片）。"""
    ds = train_loader.dataset
    base_ds, indices = ds, None
    if isinstance(ds, Subset):
        indices = ds.indices
        base_ds = ds.dataset
    all_labels = _extract_labels(base_ds, indices)

    counts = torch.bincount(torch.tensor(all_labels), minlength=num_classes).float()
    w = counts.sum() / (counts + 1.0)
    w = w / w.mean()
    return w
