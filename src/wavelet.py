"""db4 三层 2D-DWT，纯 PyTorch GPU 实现，输出 LL₃ 与 (LH₃, HL₃, HH₃)。"""
from __future__ import annotations

import numpy as np
import pywt
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# db4 滤波器系数（一次性初始化）
# ---------------------------------------------------------------------------
_DB4_LO: list[float] = [
    -0.010597, 0.032883, 0.030841, -0.187035,
    -0.027984, 0.630881, 0.714847, 0.230378,
]
_DB4_HI: list[float] = [
    -0.230378, 0.714847, -0.630881, -0.027984,
    0.187035, 0.030841, -0.032883, -0.010597,
]


def _db4_2d_filters(device: torch.device) -> torch.Tensor:
    """返回 (4, 1, 8, 8) 卷积核: [LL, LH, HL, HH] 各为 lo/hi 的外积。"""
    lo = torch.tensor(_DB4_LO, device=device, dtype=torch.float32)
    hi = torch.tensor(_DB4_HI, device=device, dtype=torch.float32)
    ll = torch.outer(lo, lo)
    lh = torch.outer(lo, hi)
    hl = torch.outer(hi, lo)
    hh = torch.outer(hi, hi)
    return torch.stack([ll, lh, hl, hh]).unsqueeze(1)  # (4, 1, 8, 8)


@torch.no_grad()
def batch_dwt_torch(gray_n1hw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """gray_n1hw: (N,1,H,W) 任意设备；自动选择 GPU 或 CPU 路径。
    返回:
      low:  (N,3,28,28) — LL₃ 复制 3 通道
      high: (N,3,28,28) — LH₃, HL₃, HH₃ 各一通道
    """
    if gray_n1hw.dim() != 4 or gray_n1hw.size(1) != 1:
        raise ValueError("expected (N,1,H,W)")

    device = gray_n1hw.device

    # GPU 路径：纯 PyTorch conv2d
    if device.type == "cuda":
        x = gray_n1hw
        filters = _db4_2d_filters(device)  # (4, 1, 8, 8)
        # 3 层分解
        for level in range(3):
            # 边界复制填充（GPU 友好，频率分离特性与 pywt periodization 一致）
            padded = F.pad(x, (3, 3, 3, 3), mode="replicate")  # (N,1,H+6,W+6)
            # 一次 conv2d 输出 4 子带
            out = F.conv2d(padded, filters, stride=2)  # (N,4,H/2,W/2)
            if level < 2:
                x = out[:, 0:1]  # 仅 LL 进入下一层
            else:
                # 第三层：分离子带
                ll  = out[:, 0:1]  # (N,1,28,28)
                lh  = out[:, 1:2]
                hl  = out[:, 2:3]
                hh  = out[:, 3:4]
        # 拼接输出
        low  = ll.expand(-1, 3, -1, -1)   # (N,3,28,28)
        high = torch.cat([lh, hl, hh], dim=1)  # (N,3,28,28)
        return low, high

    # CPU 路径：回退到 numpy + pywt（逐样本）
    xs = gray_n1hw.detach().cpu().numpy()
    lows, highs = [], []
    for i in range(xs.shape[0]):
        lo, hi = _dwt_db4_level3_np(xs[i, 0])
        lows.append(lo)
        highs.append(hi)
    low_t = torch.from_numpy(np.stack(lows)).permute(0, 3, 1, 2).to(device)
    high_t = torch.from_numpy(np.stack(highs)).permute(0, 3, 1, 2).to(device)
    return low_t, high_t


def _dwt_db4_level3_np(gray_hw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐样本 pywt 实现（CPU 回退路径）。"""
    coeffs = pywt.wavedec2(gray_hw, "db4", level=3, mode="periodization")
    c_a3, (c_h3, c_v3, c_d3) = coeffs[0], coeffs[1]
    ll = np.asarray(c_a3, dtype=np.float32)
    lh = np.asarray(c_h3, dtype=np.float32)
    hl = np.asarray(c_v3, dtype=np.float32)
    hh = np.asarray(c_d3, dtype=np.float32)
    low_3ch  = np.stack([ll, ll, ll], axis=-1)
    high_3ch = np.stack([lh, hl, hh], axis=-1)
    return low_3ch, high_3ch
