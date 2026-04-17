"""Deterministic seeding utilities."""

from __future__ import annotations

import os
import random
from typing import Optional

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - optional dependency for runtime
    np = None
import torch


def set_deterministic_seed(seed: int, deterministic: bool = True) -> None:
    """Set seeds for python, numpy, and torch.

    Args:
        seed: Random seed value.
        deterministic: If True, enforce deterministic algorithms.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only=True: log instead of crash for ops without deterministic impl
        torch.use_deterministic_algorithms(True, warn_only=True)
