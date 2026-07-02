"""
SCN (Self-Cure Network) — CVPR 2020

论文: Wang K, Peng X, Yang J, Lu S, Qiao Y.
     "Suppressing Uncertainties for Large-Scale Facial Expression Recognition." CVPR 2020.
     https://arxiv.org/abs/2002.10392

核心模块:
  1. Self-Attention Importance Weighting: FC+Sigmoid 学习每样本重要性
  2. Rank Regularization (RR-Loss): 高低重要性组的 margin-based 正则
  3. Relabeling: 低重要性样本在 max_pred > threshold 时重标

实现说明:
  - Backbone: ResNet-18 (MS-Celeb-1M 预训练 → ImageNet 替代)
  - batch_size=64, margin_1=0.07 (小 batch 设定)
  - 重标从 epoch 10 开始, margin_2=0.2
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class SelfAttentionWeighting(nn.Module):
    """自注意力重要性加权模块。"""

    def __init__(self, feat_dim: int):
        super().__init__()
        self.fc = nn.Linear(feat_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, feat_dim) — 特征向量
        Returns:
            alpha: (B,) — 重要性权重, [0,1]
        """
        return torch.sigmoid(self.fc(features)).squeeze(-1)


def rank_regularization_loss(
    alpha: torch.Tensor,
    labels: torch.Tensor,
    logits: torch.Tensor,
    margin_1: float = 0.07,
) -> torch.Tensor:
    """
    Rank Regularization Loss (RR-Loss).

    将样本按重要性降序排列, 分高/低重要性两组。
    强制高重要性组的平均损失 < 低重要性组 — 实现 margin。

    Args:
        alpha: (B,) 重要性权重
        labels: (B,) 标签
        logits: (B, C) 预测 logits
        margin_1: 高低组间隔
    Returns:
        rr_loss: 标量
    """
    device = alpha.device
    B = alpha.size(0)

    # 按 alpha 降序排列
    sorted_idx = torch.argsort(alpha, descending=True)
    mid = B // 2
    high_idx = sorted_idx[:mid]
    low_idx = sorted_idx[mid:]

    # 每组交叉熵 (逐样本, 不求均值)
    ce = F.cross_entropy(logits, labels, reduction="none")  # (B,)

    high_mean = ce[high_idx].mean()
    low_mean = ce[low_idx].mean()

    # RR-Loss: max(0, high_mean - low_mean + margin_1)
    rr_loss = torch.clamp(high_mean - low_mean + margin_1, min=0.0)

    # 额外约束: 高重要性组 alpha 均值应更大
    alpha_high_mean = alpha[high_idx].mean()
    alpha_low_mean = alpha[low_idx].mean()
    alpha_loss = torch.clamp(alpha_low_mean - alpha_high_mean + 0.1, min=0.0)

    return rr_loss + 0.1 * alpha_loss


def relabel_samples(
    logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: torch.Tensor,
    margin_2: float = 0.2,
    relabel_ratio: float = 0.25,
) -> torch.Tensor:
    """
    对低重要性样本重标。

    条件: max_prob > prob_of_given_label + margin_2

    Args:
        logits: (B, C)
        labels: (B,)
        alpha: (B,) 重要性权重
        margin_2: 重标阈值
        relabel_ratio: 考虑重标的低重要性样本比例
    Returns:
        modified_labels: (B,) — 仅低重要性满足条件者被修改
    """
    device = logits.device
    B = logits.size(0)

    probs = F.softmax(logits, dim=1)  # (B, C)
    max_prob, max_label = probs.max(dim=1)  # (B,), (B,)
    given_prob = probs[torch.arange(B, device=device), labels]  # (B,)

    # 低重要性样本
    n_low = max(1, int(B * relabel_ratio))
    _, low_idx = torch.topk(alpha, n_low, largest=False)

    # 重标条件
    condition = (max_prob > given_prob + margin_2) & (max_label != labels)

    modified = labels.clone()
    for idx in low_idx:
        if condition[idx]:
            modified[idx] = max_label[idx]

    return modified


class SCNLoss(nn.Module):
    """SCN 联合损失: CE + RR-Loss, 支持重标。"""

    def __init__(
        self,
        margin_1: float = 0.07,
        margin_2: float = 0.2,
        relabel_epoch: int = 10,
        lambda_rr: float = 0.1,
    ):
        super().__init__()
        self.margin_1 = margin_1
        self.margin_2 = margin_2
        self.relabel_epoch = relabel_epoch
        self.lambda_rr = lambda_rr

    def forward(
        self,
        logits: torch.Tensor,
        features: torch.Tensor,
        labels: torch.Tensor,
        alpha_module: SelfAttentionWeighting,
        epoch: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            total_loss, ce_loss, rr_loss, alpha
        """
        alpha = alpha_module(features)  # (B,)

        # 重标 (epoch >= relabel_epoch 时启用)
        if epoch >= self.relabel_epoch:
            labels_used = relabel_samples(logits, labels, alpha, self.margin_2)
        else:
            labels_used = labels

        # CE loss
        ce_loss = F.cross_entropy(logits, labels_used)

        # RR loss
        rr_loss = rank_regularization_loss(alpha, labels_used, logits, self.margin_1)

        total = ce_loss + self.lambda_rr * rr_loss
        return total, ce_loss, rr_loss, alpha
