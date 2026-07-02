"""共享权重双分支 ResNet-18 骨干 + RGB 基线（224）。

支持两种预训练 backbone：
  - build_backbone_224: ImageNet 预训练 torchvision ResNet-18（默认）
  - build_iresnet18_face_backbone: MS1MV3 ArcFace 预训练 iresnet18（消融实验用）
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torchvision.models as models


def _strip_fc(backbone: nn.Module) -> nn.Module:
    children = list(backbone.children())
    return nn.Sequential(*children[:-1])


def build_backbone_224(pretrained: bool = True) -> tuple[nn.Module, int]:
    w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.resnet18(weights=w)
    feat = _strip_fc(m)
    return feat, 512


def build_iresnet18_face_backbone(face_weights_path: str) -> tuple[nn.Module, int]:
    """
    构建 MS1MV3 ArcFace 预训练的 iresnet18 backbone（输出 512-d）。

    架构差异 vs 标准 ResNet-18：
      - iresnet18 使用 pre-activation block + PReLU（非 ReLU）
      - stem 为 Conv(3,64,3,1,1) 而非 Conv(7,2,3)+MaxPool
      - 总参数量与 ResNet-18 相近（~11.2M vs ~11.7M）

    Returns:
        (backbone, feature_dim) — backbone 接收 [B,3,224,224]，输出 [B,512]。
    """
    from src.iresnet import iresnet18

    full_model = iresnet18(pretrained_path=face_weights_path)

    # 提取 backbone：保留 conv1→bn1→prelu→layer1-4→bn2
    # 去除 fc 和 features（ArcFace embedding 层）
    backbone = nn.Sequential(
        full_model.conv1,
        full_model.bn1,
        full_model.prelu,
        full_model.layer1,
        full_model.layer2,
        full_model.layer3,
        full_model.layer4,
        full_model.bn2,
        nn.AdaptiveAvgPool2d(1),  # 14×14 → 1×1
        nn.Flatten(1),             # [B,512,1,1] → [B,512]
    )
    return backbone, 512


def build_backbone_28(pretrained: bool = True) -> tuple[nn.Module, int]:
    """
    首层改为 3×3 stride1，去掉 maxpool，适配 28×28。
    若使用 ImageNet 预训练，将原 7×7 卷积核中心裁剪为 3×3 以尽量对齐特征。
    """
    w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.resnet18(weights=w)
    old_conv = m.conv1
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    if pretrained and old_conv.weight.shape[-1] >= 3:
        with torch.no_grad():
            ow = old_conv.weight.data
            c0 = (ow.shape[-1] - 3) // 2
            m.conv1.weight.copy_(ow[:, :, c0 : c0 + 3, c0 : c0 + 3])
    m.maxpool = nn.Identity()
    feat = _strip_fc(m)
    return feat, 512


AblationMode = Literal["rgb", "fusion", "low_only", "high_only"]


class FERWaveletModel(nn.Module):
    """四组消融：rgb / fusion / low_only / high_only。"""

    def __init__(
        self,
        mode: AblationMode,
        num_classes: int = 7,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.mode = mode
        if mode == "rgb":
            self.backbone, dim = build_backbone_224(pretrained)
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(dim, num_classes),
            )
        else:
            self.backbone, dim = build_backbone_28(pretrained)
            hid = 256
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(dim * 2, hid),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hid, num_classes),
            )

    def forward(
        self,
        rgb: torch.Tensor | None,
        low: torch.Tensor | None,
        high: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.mode == "rgb":
            assert rgb is not None
            x = self.backbone(rgb).flatten(1)
            return self.head(x)

        assert low is not None and high is not None
        f_low = self.backbone(low).flatten(1)
        f_high = self.backbone(high).flatten(1)
        if self.mode == "fusion":
            x = torch.cat([f_low, f_high], dim=1)
        elif self.mode == "low_only":
            z = torch.zeros_like(f_high)
            x = torch.cat([f_low, z], dim=1)
        elif self.mode == "high_only":
            z = torch.zeros_like(f_low)
            x = torch.cat([z, f_high], dim=1)
        else:
            raise ValueError(self.mode)
        return self.head(x)
