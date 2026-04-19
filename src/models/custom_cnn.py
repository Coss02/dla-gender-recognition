"""Custom CNN baseline for binary gender classification.

A 4-block convolutional network trained from scratch.
Intentionally simple to serve as a lower-bound baseline
against finetuned pretrained models.

Architecture:
    4 x (Conv3x3 -> BatchNorm -> ReLU -> MaxPool2x2)
    -> AdaptiveAvgPool(1) -> FC(256,128) -> ReLU -> Dropout -> FC(128,1)
"""

import torch.nn as nn

from src.models.factory import register_model


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU -> MaxPool."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


@register_model("custom_cnn")
class CustomCNN(nn.Module):
    """4-block CNN with adaptive pooling and a two-layer classifier head.

    Args:
        config: Model config dict. Uses "dropout" key (default 0.3).
    """

    def __init__(self, config: dict):
        super().__init__()
        dropout = config.get("dropout", 0.3)

        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)
