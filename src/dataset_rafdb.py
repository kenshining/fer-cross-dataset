"""
RAF-DB 数据集：读取 train.txt/validation.txt，匹配 Training/{0-6}/ 和 PublicTest/{0-6}/ 目录。

数据目录结构:
  RAF-DB/
    train.txt          # 每行: "EmotionName\\filename.jpg" (如 "Anger\\149.jpg")
    validation.txt     # 同上格式
    Training/{0-6}/    # 各类别图片
    PublicTest/{0-6}/  # 各类别图片

标签映射: 文本类名 → 统一 0-6 索引 (angry=0, disgust=1, fear=2, happy=3, sad=4, surprise=5, neutral=6)
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


# RAF-DB 文本类名 → 统一 0-6 索引映射
# 注意: 训练/验证 txt 文件中使用的是英文类名（首字母大写）
EMOTION_NAME_TO_INDEX = {
    "Anger": 0,
    "Disgust": 1,
    "Fear": 2,
    "Happiness": 3,
    "Sadness": 4,
    "Surprise": 5,
    "Neutral": 6,
}

# 统一 0-6 索引 → 类名（用于输出可读结果）
INDEX_TO_NAME = {v: k for k, v in EMOTION_NAME_TO_INDEX.items()}


class RAFDBDataset(Dataset):
    """RAF-DB 数据集。

    参数:
        root: RAF-DB 数据集根目录（含 train.txt / validation.txt）
        split: "train" → 读 train.txt + Training/ 目录
                "test"  → 读 validation.txt + PublicTest/ 目录
    """

    def __init__(self, root: str | Path, split: str = "train"):
        root = Path(root)
        if split == "train":
            txt_path = root / "train.txt"
            img_subdir = root / "Training"
        else:
            txt_path = root / "validation.txt"
            img_subdir = root / "PublicTest"

        if not txt_path.is_file():
            raise FileNotFoundError(f"找不到 RAF-DB 标注文件: {txt_path}")

        self.samples: list[tuple[Path, int]] = []
        lines = txt_path.read_text(encoding="utf-8").strip().splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 每行格式: "EmotionName\filename.jpg"
            # 注意: txt 文件中使用反斜杠 \ 作为分隔符
            if "\\" in line:
                emotion_name, filename = line.rsplit("\\", 1)
            else:
                # 兼容 Linux 下可能出现的正斜杠
                emotion_name, filename = line.rsplit("/", 1)

            cls_idx = EMOTION_NAME_TO_INDEX.get(emotion_name)
            if cls_idx is None:
                continue  # 跳过未知类名

            img_path = img_subdir / str(cls_idx) / filename
            if not img_path.is_file():
                continue  # 跳过可能缺失的图片

            self.samples.append((img_path, cls_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        """返回 (PIL Image RGB, 标签 0-6)"""
        img_path, label = self.samples[idx]
        pil = Image.open(img_path).convert("RGB")
        return pil, label


def rafdb_collate_fn(batch: list[tuple]) -> tuple[list[Image.Image], torch.Tensor]:
    """DataLoader 批处理：PIL 列表 + 标签张量。"""
    pils = [b[0] for b in batch]
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return pils, ys
