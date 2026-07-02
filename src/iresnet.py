"""
InsightFace iresnet18 架构定义。

与 torchvision resnet18 的关键差异：
- Pre-activation 风格：BN → Conv → BN → PReLU → Conv → BN
- 每个 block 始终包含 1×1 downsample 分支
- Stem: Conv2d(3,64,3,1,1) + BN + PReLU（无 stride-2，无 MaxPool）
- 输出层: BN → Flatten → Linear → BN（ArcFace embedding）

该模块严格遵循 InsightFace MS1MV3 预训练权重的命名约定，
确保 load_state_dict(strict=True) 通过。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ImprovedResidual(nn.Module):
    """InsightFace Improved Residual Block。

    第一个 block（stride=2 或 in_ch≠out_ch）包含 downsample 分支；
    后续 block（stride=1 且 in_ch=out_ch）不含 downsample。
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.prelu = nn.PReLU(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)

        # downsample 仅在需要匹配维度或空间尺寸时存在
        if in_ch != out_ch or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.downsample(x) if self.downsample is not None else x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        return out + shortcut


class IResNet(nn.Module):
    """
    InsightFace IResNet backbone。

    层配置（如 iresnet18）：
        layers = [2, 2, 2, 2]  各 stage 的 block 数
        channels = [64, 128, 256, 512]  各 stage 的输出通道
    """

    def __init__(
        self,
        layers: list[int],
        in_channels: int = 3,
        embedding_dim: int = 512,
    ):
        super().__init__()
        self.layers = layers

        # ── Stem ──
        self.conv1 = nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU(64)

        # ── Residual stages ──
        ch = 64
        self.layer1 = self._make_layer(64, 64,  layers[0], stride=2)
        self.layer2 = self._make_layer(64, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(128, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(256, 512, layers[3], stride=2)

        # ── Output ──
        self.bn2 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512 * 7 * 7, embedding_dim)
        self.features = nn.BatchNorm1d(embedding_dim)

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [ImprovedResidual(in_ch, out_ch, stride=stride)]
        for _ in range(1, blocks):
            layers.append(ImprovedResidual(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.features(x)
        return x


def iresnet18(pretrained_path: str | None = None) -> IResNet:
    """
    创建 iresnet18 实例，可选择加载 InsightFace 预训练权重。

    Args:
        pretrained_path: MS1MV3 / Glint360K 权重的 .pth 文件路径。
                         若为 None，返回随机初始化的模型。

    Returns:
        IResNet 实例（默认输入 112×112，输出 512 维 ArcFace embedding）。
    """
    model = IResNet(layers=[2, 2, 2, 2])
    if pretrained_path is not None:
        ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt, strict=True)
    return model
