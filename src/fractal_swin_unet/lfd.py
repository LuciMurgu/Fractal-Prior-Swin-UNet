"""Local fractal dimension (LFD) map approximation utilities."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F


def _local_variance(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Compute local variance with a square window.

    Args:
        x: Tensor of shape (B, C, H, W).
        kernel_size: Window size.

    Returns:
        Tensor of shape (B, C, H, W) with local variance.
    """

    padding = kernel_size // 2
    mean = F.avg_pool2d(x, kernel_size, stride=1, padding=padding)
    mean_sq = F.avg_pool2d(x * x, kernel_size, stride=1, padding=padding)
    return torch.clamp(mean_sq - mean * mean, min=0.0)


def compute_lfd_map(
    x: torch.Tensor,
    scales: Iterable[int] | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Approximate an LFD-like map using multi-scale local variance.

    This is a deterministic, bounded proxy used for scaffolding and tests.

    Args:
        x: Input tensor of shape (B, C, H, W).
        scales: Iterable of kernel sizes. Defaults to (3, 5, 7).
        eps: Small value for numerical stability.

    Returns:
        Tensor of shape (B, 1, H, W) normalized to [0, 1].
    """

    if x.ndim != 4:
        raise ValueError("Input must be a 4D tensor (B, C, H, W).")

    if scales is None:
        scales = (3, 5, 7)

    x_gray = x.mean(dim=1, keepdim=True)
    variance_maps = [_local_variance(x_gray, k) for k in scales]
    lfd = torch.stack(variance_maps, dim=0).mean(dim=0)

    min_val = lfd.amin(dim=(2, 3), keepdim=True)
    max_val = lfd.amax(dim=(2, 3), keepdim=True)
    lfd_norm = (lfd - min_val) / (max_val - min_val + eps)
    return lfd_norm.clamp(0.0, 1.0)
