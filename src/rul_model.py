"""
RUL (Relative Uncertainty Learning) — NeurIPS 2021

论文: Zhang Y, Wang C, Deng W.
     "Relative Uncertainty Learning for Facial Expression Recognition." NeurIPS 2021.
     https://arxiv.org/abs/2106.08472

核心模块:
  1. Feature Mixup: 在特征空间混合两样本，学习相对难易程度
  2. Uncertainty Branch: 额外分支预测每个样本的不确定性
  3. Add-up Loss: 从混合特征同时识别两个表情

实现说明:
  - Backbone: ResNet-18 (ImageNet预训练)
  - 推理时移除Mixup分支，标准ResNet-18推理
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class RULModel(nn.Module):
    """RUL: ResNet-18 + Uncertainty Branch."""

    def __init__(self, num_classes: int = 7, feat_dim: int = 512):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        backbone = resnet18(weights=weights)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])  # (B,512,1,1)
        self.feat_dim = feat_dim

        # 分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        # 不确定性分支 (仅在训练时使用)
        self.uncertainty_branch = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # 不确定性 ∈ [0,1]
        )

    def forward(self, x: torch.Tensor):
        f = self.encoder(x)  # (B, 512, 1, 1)
        f_flat = f.view(f.size(0), -1)  # (B, 512)
        logits = self.classifier(f)
        uncertainty = self.uncertainty_branch(f).squeeze(-1)  # (B,)
        return logits, f_flat, uncertainty


def feature_mixup(
    f1: torch.Tensor,
    f2: torch.Tensor,
    lam: torch.Tensor,
) -> torch.Tensor:
    """特征空间mixup: f_mix = lam * f1 + (1-lam) * f2."""
    return lam * f1 + (1 - lam) * f2


def add_up_loss(
    logits: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: torch.Tensor,
) -> torch.Tensor:
    """
    Add-up Loss: 从混合特征中同时识别两个标签。

    L_add = -sum(lam * log(p(y_a|x_mix)) + (1-lam) * log(p(y_b|x_mix)))

    Args:
        logits: (2B, C) 混合特征的预测logits
        labels_a: (B,) 样本A的标签
        labels_b: (B,) 样本B的标签
        lam: (B,) mixup系数
    """
    # logits是 (2B, C): [原始batch, 混合batch]
    B = labels_a.size(0)
    logits_orig = logits[:B]
    logits_mix = logits[B:]

    # 原始样本的标准交叉熵
    ce_orig = F.cross_entropy(logits_orig, labels_a)

    # 混合样本的softmax + add-up loss
    log_probs = F.log_softmax(logits_mix, dim=1)
    loss_a = -log_probs[torch.arange(B, device=logits.device), labels_a]
    loss_b = -log_probs[torch.arange(B, device=logits.device), labels_b]

    add_up = (lam * loss_a + (1 - lam) * loss_b).mean()

    return ce_orig + add_up


def rul_uncertainty_loss(
    uncertainty_a: torch.Tensor,
    uncertainty_b: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
) -> torch.Tensor:
    """
    不确定性正则: 不确定性高的样本应有更大的损失。

    Args:
        uncertainty_a: (B,) 样本A的不确定性
        uncertainty_b: (B,) 样本B的不确定性
        labels_a, labels_b: (B,)
        logits_a, logits_b: (B, C)
    """
    ce_a = F.cross_entropy(logits_a, labels_a, reduction="none")
    ce_b = F.cross_entropy(logits_b, labels_b, reduction="none")

    # 不确定性应与损失正相关
    loss_u = -(uncertainty_a * ce_a.detach()).mean() - (uncertainty_b * ce_b.detach()).mean()

    return 0.01 * loss_u  # 小权重
