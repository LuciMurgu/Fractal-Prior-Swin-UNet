"""Lacunarity map computation via gliding-box algorithm.

Lacunarity measures the "gappiness" or heterogeneity of texture,
complementary to fractal dimension. High lacunarity = clustered/
heterogeneous structure, low = uniform distribution.

Reference: Plotnick et al. (1996) "Lacunarity analysis: A general
technique for the analysis of spatial patterns."
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F


def _to_grayscale(image: torch.Tensor) -> torch.Tensor:
    """Convert (C, H, W) or (H, W) to (H, W) float grayscale."""
    if image.ndim == 3:
        if image.shape[0] == 1:
            return image[0].float()
        return image.mean(dim=0).float()
    if image.ndim == 2:
        return image.float()
    raise ValueError("image must be a 2D or 3D tensor.")


def gliding_box_lacunarity(
    patch: torch.Tensor,
    box_sizes: Sequence[int] = (4, 8, 16),
) -> float:
    """Compute average lacunarity across multiple box sizes.

    Uses the gliding-box algorithm: for each box size r, slide an r×r
    window across the patch, compute the mass (sum of pixel values) in
    each window, then lacunarity = var(mass) / mean(mass)^2.

    Args:
        patch: Grayscale patch tensor (H, W).
        box_sizes: Sequence of box sizes to average over.

    Returns:
        Average lacunarity (float).
    """
    gray = _to_grayscale(patch).float()

    # Normalize to [0, 1]
    vmin, vmax = gray.min(), gray.max()
    if (vmax - vmin) > 0:
        gray = (gray - vmin) / (vmax - vmin)
    else:
        return 0.0

    lacunarities = []
    for r in box_sizes:
        if r <= 0 or r > min(gray.shape):
            continue
        # Use avg_pool2d to efficiently compute mass sums
        # Mass = sum of pixel values in each r×r window = r*r * avg_pool
        avg = F.avg_pool2d(
            gray.unsqueeze(0).unsqueeze(0),
            kernel_size=r,
            stride=1,
            padding=0,
        )
        mass = avg * (r * r)  # convert average back to sum

        mu = mass.mean()
        var = mass.var() if mass.numel() > 1 else torch.tensor(0.0, device=mass.device)

        if mu > 1e-8:
            lac = float((var / (mu * mu)).item()) + 1.0
        else:
            lac = 1.0
        lacunarities.append(lac)

    if not lacunarities:
        return 0.0
    return sum(lacunarities) / len(lacunarities)


def compute_lacunarity_map(
    image: torch.Tensor,
    window_size: int = 32,
    stride: Optional[int] = None,
    box_sizes: Sequence[int] = (4, 8, 16),
) -> torch.Tensor:
    """Compute a pixel-level lacunarity map via sliding windows.

    Args:
        image: Tensor of shape (C, H, W) or (H, W).
        window_size: Size of the sliding window for local computation.
        stride: Step size between windows (default: window_size // 2).
        box_sizes: Box sizes for the gliding-box algorithm within each window.

    Returns:
        Tensor of shape (H, W) with float32 values, normalized to [0, 1].
    """
    gray = _to_grayscale(image).float()
    height, width = gray.shape

    if stride is None:
        stride = max(1, window_size // 2)

    top_positions = list(range(0, max(1, height - window_size + 1), stride))
    left_positions = list(range(0, max(1, width - window_size + 1), stride))

    if not top_positions:
        top_positions = [0]
    if not left_positions:
        left_positions = [0]

    grid_h = len(top_positions)
    grid_w = len(left_positions)
    lac_grid = torch.zeros((grid_h, grid_w), dtype=torch.float32, device=gray.device)

    for i, top in enumerate(top_positions):
        for j, left in enumerate(left_positions):
            patch = gray[top : top + window_size, left : left + window_size]
            if patch.shape[0] != window_size or patch.shape[1] != window_size:
                pad_h = window_size - patch.shape[0]
                pad_w = window_size - patch.shape[1]
                patch = F.pad(patch, (0, pad_w, 0, pad_h), mode="replicate")
            lac_grid[i, j] = gliding_box_lacunarity(patch, box_sizes)

    # Upsample to original resolution
    lac_map = F.interpolate(
        lac_grid.unsqueeze(0).unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)

    # Normalize to [0, 1]
    vmin, vmax = lac_map.min(), lac_map.max()
    if (vmax - vmin) > 0:
        lac_map = (lac_map - vmin) / (vmax - vmin)
    else:
        lac_map = torch.zeros_like(lac_map)

    return lac_map.clamp(0.0, 1.0).to(torch.float32)
