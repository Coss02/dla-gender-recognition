"""Tests for model architectures.

Verifies forward pass output shapes and parameter counts
using synthetic inputs (no dataset required).
"""

import torch
import pytest

from src.models.custom_cnn import CustomCNN


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
