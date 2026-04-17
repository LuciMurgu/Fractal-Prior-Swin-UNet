"""Lacunarity map computation via gliding-box algorithm (vectorized).

Lacunarity measures the "gappiness" or heterogeneity of texture,
complementary to fractal dimension. High lacunarity = clustered/
heterogeneous structure, low = uniform distribution.

Reference: Plotnick et al. (1996) "Lacunarity analysis: A general
technique for the analysis of spatial patterns."

Vectorized implementation uses F.unfold + batched avg_pool2d to
eliminate Python-level loops over window positions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

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

    Note: No per-patch normalization — raw intensity values are used to
    preserve cross-patch comparability. Normalization should be applied
    at the dataset level via percentile clamping.

    Args:
        patch: Grayscale patch tensor (H, W), raw intensity values.
        box_sizes: Sequence of box sizes to average over.

    Returns:
        Average lacunarity (float).
    """
    gray = _to_grayscale(patch).float()

    lacunarities = []
    for r in box_sizes:
        if r <= 0 or r > min(gray.shape):
            continue
        # Use avg_pool2d to efficiently compute mass sums
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
            lac = (var / (mu * mu)).item() + 1.0
        else:
            lac = 1.0
        lacunarities.append(lac)

    if not lacunarities:
        return 0.0
    return sum(lacunarities) / len(lacunarities)


def _batched_gliding_box_lacunarity(
    patches: torch.Tensor,
    box_sizes: Sequence[int] = (4, 8, 16),
) -> torch.Tensor:
    """Vectorized lacunarity for a batch of patches.

    Args:
        patches: (N, window_size, window_size) grayscale patches.
        box_sizes: Box sizes for the gliding-box algorithm.

    Returns:
        (N,) tensor of lacunarity values.
    """
    N, H, W = patches.shape
    device = patches.device

    # Accumulate lacunarity across box sizes
    lac_sum = torch.zeros(N, device=device)
    n_valid = 0

    for r in box_sizes:
        if r <= 0 or r > min(H, W):
            continue
        n_valid += 1
        # (N, 1, H, W) → avg_pool → (N, 1, H-r+1, W-r+1)
        avg = F.avg_pool2d(
            patches.unsqueeze(1),
            kernel_size=r,
            stride=1,
            padding=0,
        )
        mass = avg * (r * r)  # (N, 1, h, w)

        # Flatten spatial dims for mean/var
        mass_flat = mass.view(N, -1)   # (N, h*w)
        mu = mass_flat.mean(dim=1)     # (N,)
        var = mass_flat.var(dim=1) if mass_flat.shape[1] > 1 else torch.zeros(N, device=device)

        # lacunarity = var/mu^2 + 1; guard against mu ≈ 0
        safe_mu = mu.clamp(min=1e-8)
        lac = var / (safe_mu * safe_mu) + 1.0
        # Zero out where mu is near zero
        lac = torch.where(mu > 1e-8, lac, torch.ones_like(lac))
        lac_sum += lac

    if n_valid == 0:
        return torch.zeros(N, device=device)
    return lac_sum / n_valid


def compute_lacunarity_map(
    image: torch.Tensor,
    window_size: int = 32,
    stride: Optional[int] = None,
    box_sizes: Sequence[int] = (4, 8, 16),
    norm_percentiles: Optional[Tuple[float, float]] = None,
) -> torch.Tensor:
    """Compute a pixel-level lacunarity map via sliding windows (vectorized).

    Uses F.unfold to extract all windows at once, then computes lacunarity
    for all windows in a single batched operation. ~20-50× faster than the
    loop-based implementation.

    Args:
        image: Tensor of shape (C, H, W) or (H, W).
        window_size: Size of the sliding window for local computation.
        stride: Step size between windows (default: window_size // 2).
        box_sizes: Box sizes for the gliding-box algorithm within each window.
        norm_percentiles: Optional (p1, p99) percentile values for dataset-level
            normalization. If provided, raw lacunarity values are clamped to
            [p1, p99] and linearly scaled to [0, 1]. If None, raw values are
            returned without normalization, preserving cross-image comparability.

    Returns:
        Tensor of shape (H, W) with float32 lacunarity values.
    """
    gray = _to_grayscale(image).float()
    height, width = gray.shape

    if stride is None:
        stride = max(1, window_size // 2)

    # Pad image so unfold covers the entire spatial extent
    pad_h = max(0, window_size - height)
    pad_w = max(0, window_size - width)
    # Also ensure we have enough for the last window
    total_h = height + pad_h
    total_w = width + pad_w
    # Compute how many extra pixels needed for stride alignment
    extra_h = (window_size - ((total_h - window_size) % stride)) % stride if total_h > window_size else 0
    extra_w = (window_size - ((total_w - window_size) % stride)) % stride if total_w > window_size else 0
    pad_h += extra_h
    pad_w += extra_w

    if pad_h > 0 or pad_w > 0:
        gray_padded = F.pad(gray.unsqueeze(0).unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect").squeeze(0).squeeze(0)
    else:
        gray_padded = gray

    padded_h, padded_w = gray_padded.shape

    # Use unfold to extract all windows: (1, 1, H, W) → (N, window_size*window_size)
    gray_4d = gray_padded.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    patches = F.unfold(gray_4d, kernel_size=window_size, stride=stride)  # (1, ws*ws, N)
    patches = patches.squeeze(0).T  # (N, ws*ws)
    patches = patches.reshape(-1, window_size, window_size)  # (N, ws, ws)

    # Compute grid dimensions
    grid_h = (padded_h - window_size) // stride + 1
    grid_w = (padded_w - window_size) // stride + 1

    # Batched lacunarity computation
    lac_values = _batched_gliding_box_lacunarity(patches, box_sizes)  # (N,)
    lac_grid = lac_values.reshape(grid_h, grid_w)

    # Upsample to original resolution
    lac_map = F.interpolate(
        lac_grid.unsqueeze(0).unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)

    # Apply dataset-level percentile normalization if provided
    if norm_percentiles is not None:
        p_low, p_high = norm_percentiles
        if p_high > p_low:
            lac_map = (lac_map - p_low) / (p_high - p_low)
            lac_map = lac_map.clamp(0.0, 1.0)
        else:
            lac_map = torch.zeros_like(lac_map)

    return lac_map.to(torch.float32)


# --- Dataset-level normalization utilities ---

def compute_dataset_percentiles(
    images: list[torch.Tensor],
    window_size: int = 32,
    stride: Optional[int] = None,
    box_sizes: Sequence[int] = (4, 8, 16),
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> Tuple[float, float]:
    """Compute dataset-level lacunarity percentiles over a list of images.

    Args:
        images: List of (C, H, W) or (H, W) tensors.
        window_size: Window size for lacunarity computation.
        stride: Optional stride override.
        box_sizes: Box sizes for gliding-box.
        p_low: Low percentile (default 1st).
        p_high: High percentile (default 99th).

    Returns:
        (p_low_value, p_high_value) tuple of float percentile values.
    """
    all_values = []
    for img in images:
        lac_map = compute_lacunarity_map(img, window_size, stride, box_sizes)
        all_values.append(lac_map.flatten())

    all_vals = torch.cat(all_values)
    low_val = float(torch.quantile(all_vals, p_low / 100.0).item())
    high_val = float(torch.quantile(all_vals, p_high / 100.0).item())
    return (low_val, high_val)


def save_percentiles(percentiles: Tuple[float, float], path: str | Path) -> None:
    """Save percentile metadata to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"p1": percentiles[0], "p99": percentiles[1]}
    path.write_text(json.dumps(data, indent=2))


def load_percentiles(path: str | Path) -> Optional[Tuple[float, float]]:
    """Load percentile metadata from JSON file, or None if not found."""
    path = Path(path)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return (float(data["p1"]), float(data["p99"]))
