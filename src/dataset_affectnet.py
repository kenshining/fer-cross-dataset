"""
AffectNet 数据集：读取 Manually_Annotated_file_lists 中的 CSV 标注文件。

数据目录结构:
  AffectNet/
    Manually_Annotated_file_lists/
      training.csv          # 414,801 条人工标注记录
      validation.csv        # 5,501 条人工标注记录
    Manually_Annotated/
      Manually_Annotated/
        Manually_Annotated_Images/
          {subdir}/{filename}   # 实际图片文件

CSV 各列:
  subDirectory_filePath: "子目录/文件名" (如 "689/abc123.jpg")
  expression: AffectNet 原始标签 (0=Neutral, 1=Happy, 2=Sad, 3=Surprise,
                                    4=Fear, 5=Disgust, 6=Anger, 7=Contempt,
                                    8=None, 9=Uncertain, 10=No-Face)

标签映射: AffectNet expression → 统一 0-6 索引
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


# AffectNet expression → 统一 0-6 索引映射
# 只映射 0-6（7 类基本表情），7(Contempt)及以上跳过
AFFECTNET_EXPR_TO_INDEX = {
    0: 6,   # Neutral  → neutral
    1: 3,   # Happy    → happy
    2: 4,   # Sad      → sad
    3: 5,   # Surprise → surprise
    4: 2,   # Fear     → fear
    5: 1,   # Disgust  → disgust
    6: 0,   # Anger    → angry
}

# 统一索引 → AffectNet 原始标签
INDEX_TO_AFFECTNET_EXPR = {v: k for k, v in AFFECTNET_EXPR_TO_INDEX.items()}


class AffectNetDataset(Dataset):
    """AffectNet 人工标注子集。

    参数:
        root: AffectNet 数据集根目录（含 Manually_Annotated/ 等子目录）
        split: "train" → 读 training.csv
               "val"   → 读 validation.csv
        max_samples: 最多读取的样本数（用于快速测试，None 表示全部读取）
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        max_samples: int | None = None,
    ):
        root = Path(root)
        csv_dir = root / "Manually_Annotated_file_lists"
        image_base = (
            root
            / "Manually_Annotated"
            / "Manually_Annotated"
            / "Manually_Annotated_Images"
        )

        if split == "train":
            csv_path = csv_dir / "training.csv"
        else:
            csv_path = csv_dir / "validation.csv"

        if not csv_path.is_file():
            raise FileNotFoundError(f"找不到 AffectNet CSV: {csv_path}")

        if not image_base.is_dir():
            raise NotADirectoryError(f"找不到 AffectNet 图片目录: {image_base}")

        self.samples: list[tuple[Path, int]] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if max_samples is not None and i >= max_samples:
                    break
                expr = int(row["expression"])
                # 跳过非基本表情（Contempt=7, None=8, Uncertain=9, No-Face=10）
                if expr not in AFFECTNET_EXPR_TO_INDEX:
                    continue
                our_label = AFFECTNET_EXPR_TO_INDEX[expr]
                subpath = row["subDirectory_filePath"].strip()
                img_path = image_base / subpath
                if not img_path.is_file():
                    continue  # 跳过可能缺失的图片
                self.samples.append((img_path, our_label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        """返回 (PIL Image RGB, 标签 0-6)"""
        img_path, label = self.samples[idx]
        pil = Image.open(img_path).convert("RGB")
        return pil, label


def affectnet_collate_fn(batch: list[tuple]) -> tuple[list[Image.Image], torch.Tensor]:
    """DataLoader 批处理：PIL 列表 + 标签张量。"""
    pils = [b[0] for b in batch]
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return pils, ys
