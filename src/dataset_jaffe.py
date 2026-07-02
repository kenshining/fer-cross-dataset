"""
JAFFE (Japanese Female Facial Expression) 数据集：用于补充跨域验证。

数据目录结构:
  Jaffe/
    anger/       # ~30 张
    disgust/     # ~29 张
    fear/        # ~33 张
    happiness/   # ~31 张
    neutral/     # ~30 张
    sadness/     # ~31 张
    surprise/    # ~30 张
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


# JAFFE 目录类名 → 统一 0-6 索引
EMOTION_NAME_TO_INDEX = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "happiness": 3,
    "sadness": 4,
    "surprise": 5,
    "neutral": 6,
}

# 统一 0-6 索引 → JAFFE 类名
INDEX_TO_NAME = {v: k for k, v in EMOTION_NAME_TO_INDEX.items()}


class JAFFEDataset(Dataset):
    """JAFFE 数据集（补充评估用）。

    参数:
        root: JAFFE 数据集根目录（含 7 个类别子目录）
        allowed_extensions: 允许的图片格式，默认 {".jpg", ".jpeg", ".png", ".tiff"}
    """

    def __init__(
        self,
        root: str | Path,
        allowed_extensions: set[str] | None = None,
    ):
        root = Path(root)
        if not root.is_dir():
            raise NotADirectoryError(f"找不到 JAFFE 目录: {root}")

        if allowed_extensions is None:
            allowed_extensions = {".jpg", ".jpeg", ".png", ".tiff"}

        self.samples: list[tuple[Path, int]] = []

        # 遍历 7 个类别子目录
        for dir_name in sorted(EMOTION_NAME_TO_INDEX.keys()):
            class_dir = root / dir_name
            if not class_dir.is_dir():
                continue
            cls_idx = EMOTION_NAME_TO_INDEX[dir_name]
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in allowed_extensions:
                    self.samples.append((img_path, cls_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        """返回 (PIL Image RGB, 标签 0-6)"""
        img_path, label = self.samples[idx]
        pil = Image.open(img_path).convert("RGB")
        return pil, label


def jaffe_collate_fn(batch: list[tuple]) -> tuple[list[Image.Image], torch.Tensor]:
    """DataLoader 批处理：PIL 列表 + 标签张量。"""
    pils = [b[0] for b in batch]
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return pils, ys
