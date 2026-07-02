"""人脸检测与 224 裁剪：优先 YOLO（若安装 ultralytics 且提供权重），否则回退中心裁剪。"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image


def center_crop_resize(rgb: Image.Image, size: int = 224) -> Image.Image:
    w, h = rgb.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    crop = rgb.crop((left, top, left + side, top + side))
    return crop.resize((size, size), Image.BICUBIC)


def yolo_crop_if_available(
    rgb: Image.Image,
    weights_path: Optional[str],
    size: int = 224,
) -> Image.Image:
    if not weights_path or not Path(weights_path).is_file():
        return center_crop_resize(rgb, size)
    try:
        from ultralytics import YOLO
    except ImportError:
        return center_crop_resize(rgb, size)
    model = YOLO(weights_path)
    arr = np.array(rgb.convert("RGB"))
    res = model.predict(arr, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return center_crop_resize(rgb, size)
    # 最大面积框
    xyxy = res.boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    idx = int(np.argmax(areas))
    x1, y1, x2, y2 = res.boxes.xyxy[idx].cpu().numpy().tolist()
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(rgb.size[0], int(x2)), min(rgb.size[1], int(y2))
    if x2 <= x1 or y2 <= y1:
        return center_crop_resize(rgb, size)
    face = rgb.crop((x1, y1, x2, y2))
    return face.resize((size, size), Image.BICUBIC)


def pil_to_tensor01(rgb224: Image.Image) -> torch.Tensor:
    t = torch.from_numpy(np.array(rgb224, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return t
