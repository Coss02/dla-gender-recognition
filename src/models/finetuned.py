"""Finetuned pretrained models for binary gender classification.

Supports MobileNetV2 (from torchvision) and Xception (from timm).
Both use a two-phase finetuning strategy:
  1. Freeze backbone, train only the new classification head (warmup)
  2. Unfreeze backbone (or last N layers), train end-to-end with low lr

The Trainer handles the training loop; this module only constructs
the models with the correct head and initial freeze state.
"""

import timm
import torch.nn as nn
from torchvision import models

from src.models.factory import register_model


def _freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters except those in the 'classifier' or 'fc' group."""
    for name, param in model.named_parameters():
        if "classifier" not in name and "fc" not in name and "head" not in name:
            param.requires_grad = False


def _count_frozen(model: nn.Module) -> tuple[int, int]:
    """Return (frozen_params, total_params) for logging."""
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return frozen, total


@register_model("mobilenet_v2")
class MobileNetV2Gender(nn.Module):
    """MobileNetV2 with replaced classification head.

    The original classifier is:
        Sequential(Dropout(0.2), Linear(1280, 1000))
    We replace it with:
        Sequential(Dropout(dropout), Linear(1280, 1))
    """

    def __init__(self, config: dict):
        super().__init__()
        pretrained = config.get("pretrained", True)
        dropout = config.get("dropout", 0.3)

        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v2(weights=weights)

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, 1),
        )

        if pretrained:
            _freeze_backbone(self.backbone)
            frozen, total = _count_frozen(self.backbone)
            print(f"MobileNetV2: froze {frozen:,}/{total:,} parameters for warmup")

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters for end-to-end finetuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


@register_model("xception")
class XceptionGender(nn.Module):
    """Xception (via timm) with replaced classification head.

    timm's Xception uses a 'fc' layer as the final classifier.
    We replace it with a single Linear(2048, 1).
    """

    def __init__(self, config: dict):
        super().__init__()
        pretrained = config.get("pretrained", True)
        dropout = config.get("dropout", 0.3)

        self.backbone = timm.create_model("legacy_xception", pretrained=pretrained)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),
        )

        if pretrained:
            _freeze_backbone(self.backbone)
            frozen, total = _count_frozen(self.backbone)
            print(f"Xception: froze {frozen:,}/{total:,} parameters for warmup")

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters for end-to-end finetuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
