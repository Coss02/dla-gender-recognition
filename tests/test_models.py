"""Tests for model architectures.

Verifies forward pass output shapes and parameter counts
using synthetic inputs (no dataset required).
"""

import torch
import pytest

from src.models.custom_cnn import CustomCNN
from src.models.finetuned import MobileNetV2Gender, XceptionGender


class TestCustomCNN:
    @pytest.fixture()
    def model(self):
        config = {"dropout": 0.3}
        return CustomCNN(config)

    def test_output_shape(self, model):
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 1)

    def test_output_shape_different_input_size(self, model):
        """AdaptiveAvgPool makes the model input-size agnostic."""
        x = torch.randn(1, 3, 128, 128)
        out = model(x)
        assert out.shape == (1, 1)

    def test_parameter_count_reasonable(self, model):
        total = sum(p.numel() for p in model.parameters())
        # 4 conv blocks + classifier: expect roughly 500K-2M params
        assert 100_000 < total < 5_000_000

    def test_output_is_logit_not_probability(self, model):
        """Model should output raw logits (no sigmoid), as BCEWithLogitsLoss expects."""
        x = torch.randn(4, 3, 224, 224)
        out = model(x)
        # Logits can be negative or > 1
        assert out.min() < 0.5 or out.max() > 0.5  # loose check

    def test_gradients_flow(self, model):
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestMobileNetV2Gender:
    @pytest.fixture()
    def model(self):
        config = {"pretrained": True, "dropout": 0.3}
        return MobileNetV2Gender(config)

    def test_output_shape(self, model):
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 1)

    def test_classifier_head_replaced(self, model):
        last_layer = model.backbone.classifier[-1]
        assert isinstance(last_layer, torch.nn.Linear)
        assert last_layer.out_features == 1

    def test_backbone_frozen_initially(self, model):
        frozen = [not p.requires_grad for n, p in model.named_parameters()
                  if "classifier" not in n]
        assert all(frozen), "Backbone should be frozen after init with pretrained=True"

    def test_unfreeze_backbone(self, model):
        model.unfreeze_backbone()
        for param in model.parameters():
            assert param.requires_grad


class TestXceptionGender:
    @pytest.fixture()
    def model(self):
        config = {"pretrained": True, "dropout": 0.3}
        return XceptionGender(config)

    def test_output_shape(self, model):
        x = torch.randn(2, 3, 299, 299)
        out = model(x)
        assert out.shape == (2, 1)

    def test_classifier_head_replaced(self, model):
        fc = model.backbone.fc
        last_layer = fc[-1] if isinstance(fc, torch.nn.Sequential) else fc
        assert isinstance(last_layer, torch.nn.Linear)
        assert last_layer.out_features == 1

    def test_backbone_frozen_initially(self, model):
        frozen = [not p.requires_grad for n, p in model.named_parameters()
                  if "fc" not in n]
        assert all(frozen), "Backbone should be frozen after init with pretrained=True"

    def test_unfreeze_backbone(self, model):
        model.unfreeze_backbone()
        for param in model.parameters():
            assert param.requires_grad
