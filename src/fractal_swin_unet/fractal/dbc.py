"""Differential Box Counting (DBC) fractal dimension estimate.

Provides both a single-patch API (backward compatible) and a fully
vectorized batched API that processes all patches in one shot on GPU.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def _prepare_grayscale(patch: torch.Tensor) -> torch.Tensor:
    if patch.ndim == 3:
        patch = patch.mean(dim=0)
    if patch.ndim != 2:
        raise ValueError("patch must be a 2D tensor or 3D (C, H, W).")
    patch = patch.float()
    min_val = patch.min()
    max_val = patch.max()
    if (max_val - min_val) > 0:
        patch = (patch - min_val) / (max_val - min_val)
    else:
        patch = torch.zeros_like(patch)
    return patch


def dbc_fractal_dimension(patch: torch.Tensor, box_sizes: Sequence[int]) -> float:
    """Estimate fractal dimension using DBC for multiple box sizes.

    Args:
        patch: Grayscale patch tensor (H, W) or (C, H, W).
        box_sizes: Sequence of spatial box sizes.

    Returns:
        Estimated fractal dimension (float).
    """

    patch = _prepare_grayscale(patch)
    height, width = patch.shape
    device = patch.device

    if len(box_sizes) < 2:
        raise ValueError("box_sizes must contain at least two sizes.")

    counts = []
    inv_sizes = []

    for size in box_sizes:
        if size <= 0:
            raise ValueError("box_sizes must be positive integers.")

        pad_h = (size - height % size) % size
        pad_w = (size - width % size) % size
        padded = F.pad(patch.unsqueeze(0).unsqueeze(0), (0, pad_w, 0, pad_h), mode="replicate")

        blocks = padded.unfold(2, size, size).unfold(3, size, size)
        # blocks shape: (1, 1, H_blocks, W_blocks, size, size)
        block_min = blocks.amin(dim=(-1, -2))
        block_max = blocks.amax(dim=(-1, -2))

        gray_box = 1.0 / size
        n_boxes = torch.floor((block_max - block_min) / gray_box) + 1.0
        counts.append(n_boxes.sum())
        inv_sizes.append(1.0 / float(size))

    x = torch.log(torch.tensor(inv_sizes, device=device))
    y = torch.log(torch.stack(counts).to(device))

    x_mean = x.mean()
    y_mean = y.mean()
    cov = ((x - x_mean) * (y - y_mean)).sum()
    var = ((x - x_mean) ** 2).sum()
    if var == 0:
        return 0.0
    return float((cov / var).item())


def dbc_fractal_dimension_batched(
    patches: torch.Tensor,
    box_sizes: Sequence[int],
) -> torch.Tensor:
    """Vectorized DBC fractal dimension for a batch of patches.

    Processes ALL patches simultaneously using batched tensor ops.
    No Python loops over patches — only over box_sizes (typically 4).

    Args:
        patches: Tensor of shape (N, H, W) — N grayscale patches.
        box_sizes: Sequence of spatial box sizes (e.g. (2, 4, 8, 16)).

    Returns:
        Tensor of shape (N,) with estimated fractal dimensions.
    """
    if patches.ndim != 3:
        raise ValueError("patches must be a 3D tensor (N, H, W).")
    if len(box_sizes) < 2:
        raise ValueError("box_sizes must contain at least two sizes.")

    N, H, W = patches.shape
    device = patches.device

    # Normalize each patch to [0, 1] independently
    flat = patches.reshape(N, -1)
    p_min = flat.amin(dim=1, keepdim=True)
    p_max = flat.amax(dim=1, keepdim=True)
    denom = (p_max - p_min).clamp(min=1e-8)
    normed = ((flat - p_min) / denom).reshape(N, H, W)

    # Add batch+channel dims: (N, 1, H, W)
    normed_4d = normed.unsqueeze(1)

    log_inv_sizes = []
    log_counts = []  # will be (len(box_sizes), N)

    for size in box_sizes:
        if size <= 0:
            raise ValueError("box_sizes must be positive integers.")

        pad_h = (size - H % size) % size
        pad_w = (size - W % size) % size
        padded = F.pad(normed_4d, (0, pad_w, 0, pad_h), mode="replicate")
        # padded: (N, 1, H', W')

        # Unfold into blocks: (N, 1, nH, nW, size, size)
        blocks = padded.unfold(2, size, size).unfold(3, size, size)

        # Min/max per block: (N, 1, nH, nW)
        block_min = blocks.amin(dim=(-1, -2))
        block_max = blocks.amax(dim=(-1, -2))

        gray_box = 1.0 / size
        n_boxes = torch.floor((block_max - block_min) / gray_box) + 1.0

        # Sum over all blocks per patch: (N,)
        counts_per_patch = n_boxes.reshape(N, -1).sum(dim=1)
        log_counts.append(torch.log(counts_per_patch.clamp(min=1e-8)))
        log_inv_sizes.append(torch.log(torch.tensor(1.0 / size, device=device)))

    # x: (S,) log of inverse box sizes
    x = torch.stack(log_inv_sizes)  # (S,)
    # y: (S, N) log of box counts
    y = torch.stack(log_counts)  # (S, N)

    # Linear regression slope per patch (vectorized)
    x_mean = x.mean()
    y_mean = y.mean(dim=0)  # (N,)
    x_centered = x - x_mean  # (S,)
    y_centered = y - y_mean.unsqueeze(0)  # (S, N)

    cov = (x_centered.unsqueeze(1) * y_centered).sum(dim=0)  # (N,)
    var = (x_centered ** 2).sum()  # scalar

    fd = torch.where(var > 0, cov / var, torch.zeros(N, device=device))
    return fd
