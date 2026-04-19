"""Evaluation metrics for binary classification.

All metric functions accept numpy arrays (predictions and targets)
and delegate to scikit-learn for correctness.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute accuracy, precision, recall, and F1 for binary classification.

    Args:
        y_true: Ground truth labels (0 or 1).
        y_pred: Predicted labels (0 or 1), already thresholded.

    Returns:
        Dictionary with metric names as keys and scores as values.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Return 2x2 confusion matrix: [[TN, FP], [FN, TP]]."""
    return confusion_matrix(y_true, y_pred, labels=[0, 1])
