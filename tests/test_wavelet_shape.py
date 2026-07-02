"""db4 三层 DWT：224×224 灰度 → 28×28 低频/高频三通道。"""
import torch

from src.wavelet import batch_dwt_torch


def test_batch_dwt_224_to_28():
    gray = torch.zeros(2, 1, 224, 224, dtype=torch.float32)
    low, high = batch_dwt_torch(gray)
    assert low.shape == (2, 3, 28, 28)
    assert high.shape == (2, 3, 28, 28)
    assert low.dtype == torch.float32
    assert high.dtype == torch.float32
