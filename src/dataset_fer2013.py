"""FER2013 CSV 数据集：读取 train.csv，解析 pixels 列，七类标签。"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# FER2013 官方 emotion 编码 0–6
FER2013_LABEL_NAMES = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")


class FER2013Dataset(Dataset):
    def __init__(self, csv_path: Path, transform=None):
        self.rows: list[tuple[np.ndarray, int]] = []
        self.transform = transform
        with open(csv_path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                if "pixels" not in row or "emotion" not in row:
                    continue
                pix = np.fromstring(row["pixels"], sep=" ", dtype=np.uint8)
                if pix.size != 48 * 48:
                    continue
                img = pix.reshape(48, 48)
                lab = int(row["emotion"])
                if lab < 0 or lab > 6:
                    continue
                self.rows.append((img, lab))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        gray48, y = self.rows[idx]
        rgb = np.stack([gray48] * 3, axis=-1)
        pil = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
        if self.transform:
            pil = self.transform(pil)
        return pil, y


def fer2013_collate_fn(batch: list[tuple]) -> tuple[list, torch.Tensor]:
    """DataLoader 批处理：PIL 列表 + 标签张量（与 train.evaluate 中逐样本 pipeline 对齐）。"""
    pils = [b[0] for b in batch]
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return pils, ys
