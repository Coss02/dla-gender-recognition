"""Seed management for reproducible experiments."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility.

    Covers Python stdlib, NumPy, and PyTorch (CPU + CUDA).
    Deterministic cuDNN is enabled at the cost of some speed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
