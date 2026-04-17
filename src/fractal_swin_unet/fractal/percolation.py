"""Percolation connectivity map for retinal vessel prior.

Computes a spatial map encoding how easily local image regions "percolate"
(form connected components). In vascular networks, regions with dense vessels
have low percolation thresholds (easily connected), while fragmented regions
have high thresholds (hard to connect).

This provides complementary information to fractal dimension (LFD) and
lacunarity: LFD measures self-similarity, lacunarity measures gap structure,
and percolation measures connectivity/fragmentation.

Reference: Stauffer & Aharony (1992) "Introduction to Percolation Theory."
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


def _connected_components_largest_ratio(binary: torch.Tensor) -> float:
    """Compute ratio of largest connected component to total foreground.

    Uses scipy.ndimage.label for connected component labeling.

    Args:
        binary: 2D binary tensor (H, W).

    Returns:
        Ratio in [0, 1]. Returns 0.0 if no foreground pixels.
    """
    fg_count = binary.sum().item()
    if fg_count < 1:
        return 0.0

    try:
        from scipy import ndimage
    except ImportError:
        # Fallback: assume one big component (conservative estimate)
        return 1.0

    np_binary = binary.cpu().numpy()
    labeled, num_features = ndimage.label(np_binary)
    if num_features == 0:
        return 0.0

    # Find largest component efficiently
    # Use bincount for O(N) component size computation
    component_sizes = torch.bincount(
        torch.from_numpy(labeled.ravel()),
        minlength=num_features + 1,
    )
    # Index 0 is background; find max among components 1..N
    if component_sizes.shape[0] <= 1:
        return 0.0
    largest = component_sizes[1:].max().item()
    return float(largest / fg_count)


def percolation_critical_threshold(
    patch: torch.Tensor,
    threshold_grid: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    spanning_ratio: float = 0.5,
) -> float:
    """Find the critical percolation threshold for a grayscale patch.

    Sweeps intensity thresholds from high to low. At each threshold,
    binarize the patch and check if the largest connected component
    spans a sufficient fraction of the foreground (percolation).

    Args:
        patch: Grayscale tensor (H, W) with values in [0, 1].
        threshold_grid: Thresholds to sweep (descending order preferred).
        spanning_ratio: Fraction of foreground that the largest component
            must cover to be considered "percolating" (default: 0.5).

    Returns:
        Critical threshold p_c in [0, 1].
        High p_c = dense/connected (percolates even at strict threshold).
        Low p_c = sparse/fragmented (percolation only at lenient threshold).
    """
    gray = _to_grayscale(patch).float()

    # Normalize to [0, 1]
    vmin, vmax = gray.min(), gray.max()
    if (vmax - vmin) > 0:
        gray = (gray - vmin) / (vmax - vmin)
    else:
        return 1.0  # uniform → no structure → maximum threshold

    # Sweep from high to low threshold
    sorted_grid = sorted(threshold_grid, reverse=True)
    for tau in sorted_grid:
        binary = (gray >= tau).float()
        ratio = _connected_components_largest_ratio(binary)
        if ratio >= spanning_ratio:
            return float(tau)

    # If no threshold produces percolation, return lowest threshold
    return float(sorted_grid[-1]) if sorted_grid else 1.0


def compute_percolation_map(
    image: torch.Tensor,
    window_size: int = 32,
    stride: Optional[int] = None,
    threshold_grid: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    spanning_ratio: float = 0.5,
) -> torch.Tensor:
    """Compute a pixel-level percolation connectivity map.

    For each local window, finds the critical percolation threshold
    p_c where the image transitions from fragmented to connected.

    Args:
        image: Tensor of shape (C, H, W) or (H, W).
        window_size: Size of sliding window for local percolation analysis.
        stride: Step between windows (default: window_size // 2).
        threshold_grid: Intensity thresholds to sweep per window.
        spanning_ratio: Component fraction threshold for percolation.

    Returns:
        Tensor of shape (H, W) with float32 values in [0, 1].
        High values = dense/connected regions (vessels percolate at strict threshold).
        Low values = sparse/fragmented regions (percolation only at lenient threshold).
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
    perc_grid = torch.zeros((grid_h, grid_w), dtype=torch.float32, device=gray.device)

    for i, top in enumerate(top_positions):
        for j, left in enumerate(left_positions):
            patch = gray[top: top + window_size, left: left + window_size]
            if patch.shape[0] != window_size or patch.shape[1] != window_size:
                pad_h = window_size - patch.shape[0]
                pad_w = window_size - patch.shape[1]
                patch = F.pad(patch, (0, pad_w, 0, pad_h), mode="replicate")
            perc_grid[i, j] = percolation_critical_threshold(
                patch, threshold_grid, spanning_ratio
            )

    # Upsample to original resolution
    perc_map = F.interpolate(
        perc_grid.unsqueeze(0).unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)

    # Normalize to [0, 1]
    vmin, vmax = perc_map.min(), perc_map.max()
    if (vmax - vmin) > 0:
        perc_map = (perc_map - vmin) / (vmax - vmin)
    else:
        perc_map = torch.zeros_like(perc_map)

    return perc_map.clamp(0.0, 1.0).to(torch.float32)
