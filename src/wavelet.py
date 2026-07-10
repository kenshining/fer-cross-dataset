"""可配置小波基的三层 2D-DWT，纯 PyTorch GPU + numpy/pywt CPU 双路径。"""
from __future__ import annotations

import numpy as np
import pywt
import torch
import torch.nn.functional as F

# 预计算常用小波基的 GPU 滤波器（惰性缓存）
_FILTER_CACHE: dict[str, torch.Tensor] = {}


def _get_filters(wavelet_name: str, device: torch.device) -> torch.Tensor:
    """获取 (4, 1, L, L) 卷积核: [LL, LH, HL, HH] 各为 lo/hi 的外积。
    从 pywt 动态获取滤波器系数并缓存。
    """
    cache_key = f"{wavelet_name}_{device}"
    if cache_key in _FILTER_CACHE:
        return _FILTER_CACHE[cache_key]

    w = pywt.Wavelet(wavelet_name)
    # pywt 的 dec_lo/dec_hi 是分解滤波器，用于卷积
    # 注意 pywt 的系数顺序需要反转以匹配 torch conv2d
    lo_raw = w.dec_lo[::-1].copy()
    hi_raw = w.dec_hi[::-1].copy()

    lo = torch.tensor(lo_raw, device=device, dtype=torch.float32)
    hi = torch.tensor(hi_raw, device=device, dtype=torch.float32)

    ll = torch.outer(lo, lo)
    lh = torch.outer(lo, hi)
    hl = torch.outer(hi, lo)
    hh = torch.outer(hi, hi)

    filters = torch.stack([ll, lh, hl, hh]).unsqueeze(1)  # (4, 1, L, L)
    _FILTER_CACHE[cache_key] = filters
    return filters


def _pad_for_filter(filter_len: int) -> int:
    """计算每侧 padding 量。"""
    return (filter_len - 1) // 2


@torch.no_grad()
def batch_dwt_torch(
    gray_n1hw: torch.Tensor,
    wavelet: str = "db4",
) -> tuple[torch.Tensor, torch.Tensor]:
    """gray_n1hw: (N,1,H,W) 任意设备。
    返回:
      low:  (N,3,H8,W8)  — LL₃ 复制 3 通道
      high: (N,3,H8,W8)  — LH₃, HL₃, HH₃ 各一通道
    """
    if gray_n1hw.dim() != 4 or gray_n1hw.size(1) != 1:
        raise ValueError("expected (N,1,H,W)")

    device = gray_n1hw.device

    if device.type == "cuda":
        x = gray_n1hw
        filters = _get_filters(wavelet, device)  # (4, 1, L, L)
        filter_len = filters.shape[-1]
        pad = _pad_for_filter(filter_len)

        for level in range(3):
            padded = F.pad(x, (pad, pad, pad, pad), mode="replicate")
            out = F.conv2d(padded, filters, stride=2)  # (N,4,H/2,W/2)
            if level < 2:
                x = out[:, 0:1]  # LL 进入下一层
            else:
                ll = out[:, 0:1]
                lh = out[:, 1:2]
                hl = out[:, 2:3]
                hh = out[:, 3:4]

        low = ll.expand(-1, 3, -1, -1)
        high = torch.cat([lh, hl, hh], dim=1)
        return low, high

    # CPU 路径：pywt（支持所有小波基）
    xs = gray_n1hw.detach().cpu().numpy()
    lows, highs = [], []
    for i in range(xs.shape[0]):
        lo, hi = _dwt_level3_np(xs[i, 0], wavelet)
        lows.append(lo)
        highs.append(hi)
    low_t = torch.from_numpy(np.stack(lows)).permute(0, 3, 1, 2).to(device)
    high_t = torch.from_numpy(np.stack(highs)).permute(0, 3, 1, 2).to(device)
    return low_t, high_t


def _dwt_level3_np(
    gray_hw: np.ndarray, wavelet: str = "db4"
) -> tuple[np.ndarray, np.ndarray]:
    """逐样本 pywt 3-level DWT。"""
    coeffs = pywt.wavedec2(gray_hw, wavelet, level=3, mode="periodization")
    c_a3, (c_h3, c_v3, c_d3) = coeffs[0], coeffs[1]
    ll = np.asarray(c_a3, dtype=np.float32)
    lh = np.asarray(c_h3, dtype=np.float32)
    hl = np.asarray(c_v3, dtype=np.float32)
    hh = np.asarray(c_d3, dtype=np.float32)
    low_3ch = np.stack([ll, ll, ll], axis=-1)
    high_3ch = np.stack([lh, hl, hh], axis=-1)
    return low_3ch, high_3ch
