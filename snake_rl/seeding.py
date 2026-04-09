from __future__ import annotations

import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic_torch: bool = False) -> None:
    """Seed python, numpy, and torch RNGs for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch and torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
