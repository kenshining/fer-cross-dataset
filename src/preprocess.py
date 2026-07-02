"""从 PIL RGB 生成 rgb 张量、灰度与小波子带张量。"""
from __future__ import annotations

from typing import Optional

import torch
from PIL import Image

from .detect import center_crop_resize, pil_to_tensor01, yolo_crop_if_available
from .wavelet import batch_dwt_torch


def pipeline_tensors(
    pil: Image.Image,
    face_size: int,
    yolo_weights: Optional[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    返回:
      rgb_224: (1,3,224,224) [0,1]
      low: (1,3,28,28)
      high: (1,3,28,28)
    """
    crop = yolo_crop_if_available(pil.convert("RGB"), yolo_weights, face_size)
    rgb = pil_to_tensor01(crop).unsqueeze(0).to(device)
    gray = rgb.mean(dim=1, keepdim=True)
    low, high = batch_dwt_torch(gray)
    return rgb, low, high
