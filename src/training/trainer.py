"""Generic training loop for binary classification models.

The Trainer handles epoch iteration, validation, early stopping,
checkpoint saving, and local history logging. It is model-agnostic:
any nn.Module with a single sigmoid output works.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.metrics import compute_binary_metrics


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float | None = None
        self.counter = 0

    def should_stop(self, val_loss: float) -> bool:
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class Trainer:
    """Epoch-based training loop with validation, checkpointing, and history logging.

    Args:
        model: PyTorch model with single sigmoid output.
        optimizer: Configured optimizer.
        scheduler: Optional LR scheduler (stepped per epoch).
        device: Target device string ("cuda", "mps", "cpu").
        config: Full config dict saved in checkpoints for reproducibility.
        checkpoint_dir: Where to save model checkpoints.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None,
        device: str,
        config: dict[str, Any],
        checkpoint_dir: str | Path = "checkpoints",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = nn.BCEWithLogitsLoss()
        self.best_val_loss = float("inf")

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        early_stopping_patience: int = 5,
    ) -> dict[str, list[float]]:
        """Run the full training loop.

        Returns:
            History dict with keys: train_loss, val_loss, val_accuracy, etc.
        """
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
            "learning_rate": [],
        }

        for epoch in range(1, epochs + 1):
            train_loss = self._train_one_epoch(train_loader, epoch, epochs)
            val_loss, val_metrics = self._validate(val_loader)

            current_lr = self.optimizer.param_groups[0]["lr"]
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["val_precision"].append(val_metrics["precision"])
            history["val_recall"].append(val_metrics["recall"])
            history["val_f1"].append(val_metrics["f1"])
            history["learning_rate"].append(current_lr)

            print(
                f"Epoch {epoch}/{epochs} — "
                f"train_loss: {train_loss:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"val_acc: {val_metrics['accuracy']:.4f} | "
                f"val_f1: {val_metrics['f1']:.4f} | "
                f"lr: {current_lr:.6f}"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss, val_metrics)

            if self.scheduler is not None:
                self.scheduler.step()

            if early_stopping.should_stop(val_loss):
                print(f"Early stopping at epoch {epoch}")
                break

        self._save_history(history)
        return history

    def _train_one_epoch(
        self, loader: DataLoader, epoch: int, total_epochs: int
    ) -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        running_loss = 0.0
        n_batches = 0

        progress = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [train]", leave=False)
        for images, labels in progress:
            images = images.to(self.device)
            labels = labels.float().to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images).squeeze(1)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        return running_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> tuple[float, dict[str, float]]:
        """Run validation. Returns (avg_loss, metrics_dict)."""
        self.model.eval()
        running_loss = 0.0
        n_batches = 0
        all_preds: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.float().to(self.device)

            logits = self.model(images).squeeze(1)
            loss = self.criterion(logits, labels)

            running_loss += loss.item()
            n_batches += 1

            preds = (torch.sigmoid(logits) >= 0.5).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy().astype(int))

        avg_loss = running_loss / max(n_batches, 1)
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)
        metrics = compute_binary_metrics(y_true, y_pred)

        return avg_loss, metrics

    def _save_checkpoint(
        self, epoch: int, val_loss: float, val_metrics: dict[str, float]
    ) -> None:
        """Save model state, optimizer state, and metadata."""
        path = self.checkpoint_dir / "best_model.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "config": self.config,
            },
            path,
        )
        print(f"  -> Saved best checkpoint (val_loss={val_loss:.4f})")

    def _save_history(self, history: dict[str, list[float]]) -> None:
        """Persist training history as JSON for offline analysis."""
        import json

        model_name = self.config.get("model", {}).get("name", "model")
        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{model_name}_history.json"
        with open(path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"Training history saved to {path}")
