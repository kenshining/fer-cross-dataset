"""
CK+ (Extended Cohn-Kanade) 数据集：用于跨域评估。

数据目录结构:
  CK+/
    cohn-kanade-images/      # 序列图片
      {Sxxx}/{seq}/          # 如 S005/001/S005_001_00000016.png
    Emotion_labels/          # 表情标签
      {Sxxx}/{seq}/Sxxx_seq_XXXXXXXX_emotion.txt  # 文件内为 float 标签值

CK+ 原始标签:
  0=neutral, 1=anger, 2=contempt(跳过), 3=disgust, 4=fear,
  5=happy, 6=sadness, 7=surprise

协议: 取每个表情序列的倒数 N 帧（peak frames，表情峰值帧）作为正样本。
"""
from __future__ import annotations

import re
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


# CK+ 标签映射 → 统一 0-6 索引
# 跳过 contempt=2
CK_EXPR_TO_INDEX = {
    0: 6,  # neutral  → neutral
    1: 0,  # anger    → angry
    3: 1,  # disgust  → disgust
    4: 2,  # fear     → fear
    5: 3,  # happy    → happy
    6: 4,  # sadness  → sad
    7: 5,  # surprise → surprise
}

# 统一索引 → CK+ 原始标签
INDEX_TO_CK_EXPR = {v: k for k, v in CK_EXPR_TO_INDEX.items()}


class CKPlusDataset(Dataset):
    """CK+ 数据集（跨域评估用，仅提供测试/评估模式）。

    参数:
        root: CK+ 数据集根目录（含 cohn-kanade-images/ 和 Emotion_labels/）
        peak_frames: 每个序列取最后 N 帧作为表情峰值帧，默认 3
        min_frames: 序列包含的最小帧数，少于该值的序列将被跳过，默认 3
    """

    def __init__(self, root: str | Path, peak_frames: int = 3, min_frames: int = 3):
        root = Path(root)
        image_root = root / "cohn-kanade-images"
        label_root = root / "Emotion_labels"

        if not label_root.is_dir():
            raise NotADirectoryError(f"找不到 CK+ 标签目录: {label_root}")
        if not image_root.is_dir():
            raise NotADirectoryError(f"找不到 CK+ 图片目录: {image_root}")

        self.samples: list[tuple[Path, int]] = []

        # 遍历 Emotion_labels 下所有 *_emotion.txt 文件
        for label_path in sorted(label_root.rglob("*_emotion.txt")):
            # 解析标签值
            label_val = float(label_path.read_text(encoding="utf-8").strip())
            ck_label = int(round(label_val))

            # 跳过 contempt 和非基本表情
            if ck_label not in CK_EXPR_TO_INDEX:
                continue
            our_label = CK_EXPR_TO_INDEX[ck_label]

            # 根据标签文件路径推导对应的图片序列路径
            # 标签路径: Emotion_labels/S005/001/S005_001_00000016_emotion.txt
            # 图片路径: cohn-kanade-images/S005/001/S005_001_00000016.png
            rel_path = label_path.relative_to(label_root)
            subject = rel_path.parts[0]   # S005
            sequence = rel_path.parts[1]  # 001

            # 找到对应的图片序列目录
            seq_image_dir = image_root / subject / sequence
            if not seq_image_dir.is_dir():
                continue

            # 获取该序列所有图片（按文件名排序）
            all_images = sorted(seq_image_dir.iterdir())
            valid_images = [p for p in all_images if p.suffix.lower() in (".png", ".jpg", ".jpeg")]

            if len(valid_images) < min_frames:
                continue

            # 取最后 peak_frames 帧作为表情峰值帧
            peak = valid_images[-peak_frames:]
            for img_path in peak:
                self.samples.append((img_path, our_label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        """返回 (PIL Image RGB, 标签 0-6)"""
        img_path, label = self.samples[idx]
        pil = Image.open(img_path).convert("RGB")
        return pil, label


def ckplus_collate_fn(batch: list[tuple]) -> tuple[list[Image.Image], torch.Tensor]:
    """DataLoader 批处理：PIL 列表 + 标签张量。"""
    pils = [b[0] for b in batch]
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return pils, ys
