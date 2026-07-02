"""
频率域振幅扰动 (FreqCLR 核心模块)

参考:
  FACT (Xu et al., CVPR 2021) — 傅里叶振幅交换实现域泛化
  APA (Salehnia et al., IEEE MLSP 2025) — 振幅-相位增强

原理:
  img → FFT → 振幅 A + 相位 P
  扰动 A ← A × (1 + ε),  ε ~ U(-σ, σ)   (仅扰动振幅, 保留相位)
  → IFFT → 增强图像
  → 注入 RGB: 用增强的灰度差异调制原 RGB

设计要点:
  - 仅扰动振幅: 振幅编码纹理/风格/域特征
  - 保留相位: 相位编码结构(人脸身份、表情形态)
  - 自适应 σ: 小方差数据集(JAFFE)用大σ, 大方差数据集(AffectNet)用小σ
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class FreqPerturb:
    """频率域振幅扰动器。

    Args:
        sigma: 扰动强度 (0.05~0.3)
        p: 应用扰动的概率 (训练时建议 0.8~1.0)
    """

    def __init__(self, sigma: float = 0.15, p: float = 1.0):
        self.sigma = sigma
        self.p = p

    def __call__(self, rgb_batch: torch.Tensor) -> torch.Tensor:
        """对一批 RGB 图像做频率扰动。

        Args:
            rgb_batch: (B, 3, H, W), 范围 [0, 1] 或任意

        Returns:
            perturbed: (B, 3, H, W), 同范围
        """
        if self.p < 1.0 and torch.rand(1).item() > self.p:
            return rgb_batch

        return freq_perturb_batch(rgb_batch, self.sigma)

    def __repr__(self) -> str:
        return f"FreqPerturb(sigma={self.sigma}, p={self.p})"


def freq_perturb_batch(
    rgb_batch: torch.Tensor,
    sigma: float = 0.15,
) -> torch.Tensor:
    """对 batch RGB 图像做频率域振幅扰动。

    流程:
      1. RGB → Gray (保留亮度变化)
      2. Gray → FFT → 振幅 A_fft, 相位 P_fft
      3. A_fft ← A_fft × (1 + noise),  noise ~ N(0, σ²) per image
      4. IFFT → 扰动灰度
      5. 计算扰动前后的灰度比值, 应用到 RGB 三通道

    优势: 保留相位 = 保留人脸结构和表情形态;
          扰动振幅 = 模拟不同光照、纹理、图像质量

    Args:
        rgb_batch: (B, 3, H, W), 任意设备
        sigma: 扰动标准差

    Returns:
        perturbed: (B, 3, H, W), 同设备和范围
    """
    B, C, H, W = rgb_batch.shape
    device = rgb_batch.device

    # 1. RGB → Gray (ITU-R BT.601)
    gray = (
        0.299 * rgb_batch[:, 0:1] + 0.587 * rgb_batch[:, 1:2] + 0.114 * rgb_batch[:, 2:3]
    )  # (B, 1, H, W)

    # 2. FFT
    gray_fft = torch.fft.fft2(gray, dim=(-2, -1))  # (B, 1, H, W), complex
    gray_fft = torch.fft.fftshift(gray_fft, dim=(-2, -1))

    # 分离振幅和相位
    amp = torch.abs(gray_fft)  # (B, 1, H, W)
    phase = torch.angle(gray_fft)  # (B, 1, H, W)

    # 3. 对振幅施加逐张图片的随机扰动
    # noise shape: (B, 1, 1, 1), 每张图一个扰动因子
    noise = torch.randn(B, 1, 1, 1, device=device) * sigma
    amp_perturbed = amp * (1.0 + noise)

    # 4. IFFT 重建
    gray_fft_perturbed = amp_perturbed * torch.exp(1j * phase)
    gray_fft_perturbed = torch.fft.ifftshift(gray_fft_perturbed, dim=(-2, -1))
    gray_perturbed = torch.fft.ifft2(gray_fft_perturbed, dim=(-2, -1)).real

    # 5. 计算灰度变化比值, 应用到 RGB
    # ratio = gray_perturbed / (gray + eps), clamp 防止极端值
    eps = 1e-6
    ratio = gray_perturbed / (gray + eps)
    ratio = torch.clamp(ratio, 0.5, 2.0)  # 防止过强扰动

    rgb_perturbed = rgb_batch * ratio

    # 保持值域
    rgb_perturbed = torch.clamp(rgb_perturbed, 0.0, 1.0)

    return rgb_perturbed


def freq_perturb_multi_view(
    rgb_batch: torch.Tensor,
    sigma: float = 0.15,
    n_views: int = 2,
) -> list[torch.Tensor]:
    """生成同一图像的多个频率扰动视角（用于对比学习）。

    Args:
        rgb_batch: (B, 3, H, W)
        sigma: 扰动标准差
        n_views: 视角数 (包含原始图像)

    Returns:
        views: list of (B, 3, H, W), 第一个为原始图像
    """
    views = [rgb_batch]
    for _ in range(n_views - 1):
        views.append(freq_perturb_batch(rgb_batch, sigma))
    return views
