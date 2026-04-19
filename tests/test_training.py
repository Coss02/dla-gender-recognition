"""Tests for training infrastructure: metrics, trainer, and checkpointing.

Uses synthetic data to avoid dependency on real CelebA downloads.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.metrics import compute_binary_metrics, compute_confusion_matrix
from src.training.trainer import EarlyStopping, Trainer


class TestMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        metrics = compute_binary_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0

    def test_all_wrong(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        metrics = compute_binary_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 0.0

    def test_confusion_matrix_shape(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1])
        cm = compute_confusion_matrix(y_true, y_pred)
        assert cm.shape == (2, 2)

    def test_confusion_matrix_values(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        cm = compute_confusion_matrix(y_true, y_pred)
        assert cm[0, 0] == 1  # TN
        assert cm[0, 1] == 1  # FP
        assert cm[1, 0] == 1  # FN
        assert cm[1, 1] == 1  # TP


class TestEarlyStopping:
    def test_no_stop_when_improving(self):
        es = EarlyStopping(patience=3)
        assert not es.should_stop(1.0)
        assert not es.should_stop(0.9)
        assert not es.should_stop(0.8)

    def test_stops_after_patience(self):
        es = EarlyStopping(patience=2)
        es.should_stop(0.5)  # best
        es.should_stop(0.6)  # worse, counter=1
        assert es.should_stop(0.7)  # worse, counter=2 -> stop

    def test_resets_on_improvement(self):
        es = EarlyStopping(patience=2)
        es.should_stop(0.5)
        es.should_stop(0.6)  # counter=1
        es.should_stop(0.4)  # improves, counter=0
        assert not es.should_stop(0.5)  # counter=1, not yet


class _TinyModel(nn.Module):
    """Minimal model for trainer tests."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        return self.fc(x)


def _make_synthetic_loader(n_samples=32, n_features=4, batch_size=8):
    """Create a DataLoader with random features and binary labels."""
    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, 2, (n_samples,))
    return DataLoader(TensorDataset(X, y), batch_size=batch_size)


class TestTrainer:
    def test_fit_runs_one_epoch(self, tmp_path):
        model = _TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        config = {
            "wandb": {"enabled": False},
            "model": {"name": "test"},
        }

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            device="cpu",
            config=config,
            checkpoint_dir=tmp_path / "ckpts",
        )

        train_loader = _make_synthetic_loader()
        val_loader = _make_synthetic_loader(n_samples=16)

        history = trainer.fit(
            train_loader, val_loader, epochs=1, early_stopping_patience=99
        )

        assert len(history["train_loss"]) == 1
        assert len(history["val_loss"]) == 1
        assert 0.0 <= history["val_accuracy"][0] <= 1.0

    def test_checkpoint_saved(self, tmp_path):
        model = _TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        config = {
            "wandb": {"enabled": False},
            "model": {"name": "test"},
        }

        ckpt_dir = tmp_path / "ckpts"
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            device="cpu",
            config=config,
            checkpoint_dir=ckpt_dir,
        )

        train_loader = _make_synthetic_loader()
        val_loader = _make_synthetic_loader(n_samples=16)
        trainer.fit(train_loader, val_loader, epochs=2, early_stopping_patience=99)

        checkpoint_path = ckpt_dir / "best_model.pt"
        assert checkpoint_path.exists()

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in ckpt
        assert "optimizer_state_dict" in ckpt
        assert "val_loss" in ckpt

    def test_checkpoint_loads_correctly(self, tmp_path):
        model = _TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        config = {
            "wandb": {"enabled": False},
            "model": {"name": "test"},
        }

        ckpt_dir = tmp_path / "ckpts"
        trainer = Trainer(
            model=model, optimizer=optimizer, scheduler=None,
            device="cpu", config=config, checkpoint_dir=ckpt_dir,
        )
        train_loader = _make_synthetic_loader()
        val_loader = _make_synthetic_loader(n_samples=16)
        trainer.fit(train_loader, val_loader, epochs=1, early_stopping_patience=99)

        ckpt = torch.load(ckpt_dir / "best_model.pt", map_location="cpu", weights_only=False)
        new_model = _TinyModel()
        new_model.load_state_dict(ckpt["model_state_dict"])

        # Verify weights match
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.equal(p1.cpu(), p2.cpu())
