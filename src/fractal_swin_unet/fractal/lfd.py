"""Local fractal dimension (LFD) map generator.

Fully vectorized — no Python loops over patches. All DBC computations
run as a single batched GPU operation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from .cache import LFDCache
from .dbc import dbc_fractal_dimension, dbc_fractal_dimension_batched


@dataclass(frozen=True)
class LFDParams:
    """Parameters controlling LFD map generation."""

    window_size: int = 32
    stride: Optional[int] = None
    box_sizes: Sequence[int] = (2, 4, 8, 16)
    fast_mode: bool = True

    def to_hash(self) -> int:
        return hash((self.window_size, self.stride, tuple(self.box_sizes), self.fast_mode))


def _to_grayscale(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 3:
        if image.shape[0] == 1:
            return image[0]
        return image.mean(dim=0)
    if image.ndim == 2:
        return image
    raise ValueError("image must be a 2D or 3D tensor.")


def compute_lfd_map(
    image: torch.Tensor,
    params: LFDParams | None = None,
    cache: Optional[LFDCache] = None,
    image_id: Optional[str] = None,
) -> torch.Tensor:
    """Compute a normalized LFD map for an image.

    Fully vectorized: extracts all windows at once via unfold, then
    computes DBC for all patches in a single batched operation.

    Args:
        image: Tensor of shape (H, W) or (C, H, W).
        params: LFDParams controlling window/stride/box sizes.
        cache: Optional LFDCache instance.
        image_id: Optional identifier for caching.

    Returns:
        Tensor of shape (H, W) with float32 values in [0, 1].
    """

    if params is None:
        params = LFDParams()

    if cache is not None and image_id is not None:
        cached = cache.get(image_id, params.to_hash())
        if cached is not None:
            return cached

    gray = _to_grayscale(image).float()
    height, width = gray.shape

    window_size = params.window_size
    if window_size <= 0:
        raise ValueError("window_size must be positive.")

    if params.fast_mode:
        stride = params.stride if params.stride is not None else max(1, window_size // 2)
    else:
        stride = params.stride if params.stride is not None else 1

    if stride <= 0:
        raise ValueError("stride must be positive.")

    # Pad image so unfold covers the full spatial extent
    pad_h = (window_size - height % stride) % stride if height < window_size else 0
    pad_w = (window_size - width % stride) % stride if width < window_size else 0
    if pad_h > 0 or pad_w > 0:
        gray = F.pad(gray, (0, pad_w, 0, pad_h), mode="replicate")

    # Extract ALL patches at once via unfold — no Python loop
    # gray shape: (H', W')
    # patches shape: (nH, nW, window_size, window_size)
    patches = gray.unfold(0, window_size, stride).unfold(1, window_size, stride)
    grid_h, grid_w = patches.shape[0], patches.shape[1]

    # Reshape to batch: (N, window_size, window_size)
    patches_flat = patches.reshape(-1, window_size, window_size)

    # Single vectorized DBC call for ALL patches
    fd_values = dbc_fractal_dimension_batched(patches_flat, params.box_sizes)

    # Reshape back to grid
    lfd_grid = fd_values.reshape(grid_h, grid_w)

    # Upsample to original image dimensions
    lfd_grid = lfd_grid.unsqueeze(0).unsqueeze(0)
    lfd_map = F.interpolate(lfd_grid, size=(height, width), mode="bilinear", align_corners=False)
    lfd_map = lfd_map.squeeze(0).squeeze(0)

    # Normalize to [0, 1]
    min_val = lfd_map.min()
    max_val = lfd_map.max()
    if (max_val - min_val) > 0:
        lfd_map = (lfd_map - min_val) / (max_val - min_val)
    else:
        lfd_map = torch.zeros_like(lfd_map)

    lfd_map = lfd_map.clamp(0.0, 1.0).to(torch.float32)

    if cache is not None and image_id is not None:
        cache.set(image_id, params.to_hash(), lfd_map)

    return lfd_map


def compute_multiscale_ffm(
    image: torch.Tensor,
    window_sizes: Sequence[int] = (16, 32, 64),
    box_sizes: Sequence[int] = (2, 4, 8, 16),
    fast_mode: bool = True,
    include_lacunarity: bool = True,
) -> torch.Tensor:
    """Compute multi-scale Fractal Feature Maps (FFMs).

    Follows the ECCV 2024 FFM approach: compute LFD at multiple window
    sizes to capture fractal structure at different spatial scales.
    Optionally appends a lacunarity channel for texture heterogeneity.

    Args:
        image: Tensor of shape (C, H, W) or (H, W).
        window_sizes: List of window sizes for multi-scale LFD.
        box_sizes: Box sizes for DBC computation.
        fast_mode: Use strided windows for speed.
        include_lacunarity: If True, append a lacunarity channel.

    Returns:
        Tensor of shape (N, H, W) where N = len(window_sizes) + (1 if lacunarity).
    """
    channels = []
    for ws in window_sizes:
        params = LFDParams(
            window_size=ws,
            box_sizes=box_sizes,
            fast_mode=fast_mode,
        )
        lfd = compute_lfd_map(image, params=params)
        channels.append(lfd)

    if include_lacunarity:
        from .lacunarity import compute_lacunarity_map
        lac = compute_lacunarity_map(
            image,
            window_size=window_sizes[0] if window_sizes else 32,
            box_sizes=box_sizes[:3] if len(box_sizes) >= 3 else box_sizes,
        )
        channels.append(lac)

    return torch.stack(channels, dim=0)
