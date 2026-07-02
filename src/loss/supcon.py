"""
监督对比损失 (Supervised Contrastive Loss)

参考:
  Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020
  https://arxiv.org/abs/2004.11362

与 SimCLR 区别: 利用标签信息, 同类样本互为正例, 异类互为负例.
与 Triplet Loss 区别: 不依赖 hard negative mining, 训练更稳定.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (NeurIPS 2020).

    Args:
        temperature: 温度系数, 控制分布锐度 (默认 0.07)
        base_temperature: 基础温度 (默认 0.07)
        contrast_mode: 'one' = 一个 anchor vs 所有; 'all' = 所有对
    """

    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        contrast_mode: str = "all",
    ):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.contrast_mode = contrast_mode

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B * n_views, dim) — 所有视图的特征堆叠
                      例如: [img1_view1, img1_view2, img2_view1, img2_view2, ...]
            labels: (B,) — 原始标签 (不含视图复制)
        Returns:
            loss: 标量
        """
        device = features.device
        batch_size = labels.shape[0]

        # 推断 n_views
        total = features.shape[0]
        if total % batch_size != 0:
            raise ValueError(
                f"features dim 0 ({total}) 不能被 batch_size ({batch_size}) 整除, "
                f"无法推断 n_views"
            )
        n_views = total // batch_size

        # 复制标签以匹配视图数
        labels = labels.contiguous().view(-1, 1)  # (B, 1)
        mask = torch.eq(labels, labels.T).float().to(device)  # (B, B) — 同类=1

        # 扩展到多视图
        # mask_all: (n_views * B, n_views * B)
        mask_all = mask.repeat(n_views, n_views)

        # 排除自身 (对角线)
        logits_mask = torch.scatter(
            torch.ones_like(mask_all),
            1,
            torch.arange(total, device=device).view(-1, 1),
            0,
        )
        mask_all = mask_all * logits_mask

        # 计算相似度矩阵
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature,
        )

        # 数值稳定性: 减去每行最大值
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 分母: sum(exp(logits) * logits_mask)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # 分子: 同类样本的 log_prob 均值
        mean_log_prob_pos = (mask_all * log_prob).sum(1) / (mask_all.sum(1) + 1e-8)

        # 损失
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(n_views, batch_size).mean()

        return loss
