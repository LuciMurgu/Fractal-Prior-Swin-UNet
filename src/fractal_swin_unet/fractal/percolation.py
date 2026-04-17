"""Percolation connectivity map for retinal vessel prior.

Computes a spatial map encoding how easily local image regions "percolate"
(form connected components that span the lattice edge-to-edge). In vascular
networks, regions with dense vessels have high percolation thresholds
(spanning clusters form even at strict thresholds), while fragmented regions
have low thresholds (spanning only at lenient thresholds).

This provides complementary information to fractal dimension (LFD) and
lacunarity: LFD measures self-similarity, lacunarity measures gap structure,
and percolation measures connectivity/fragmentation.

Reference: Stauffer & Aharony (1992) "Introduction to Percolation Theory."
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


# ── Connected-component labeling ──────────────────────────────────────────

def _cc_label_scipy(binary: torch.Tensor) -> torch.Tensor:
    """CPU-based CC labeling via scipy.ndimage.label.

    Args:
        binary: 2D binary tensor (H, W), float or bool.

    Returns:
        Integer label tensor (H, W), 0 = background, 1..N = components.
    """
    try:
        from scipy import ndimage
    except ImportError:
        # Fallback: all foreground is one component
        return (binary > 0).long()

    np_binary = binary.cpu().numpy().astype("uint8")
    labeled, _ = ndimage.label(np_binary)
    return torch.from_numpy(labeled).long().to(binary.device)


def _cc_label_gpu(binary: torch.Tensor) -> torch.Tensor:
    """GPU-native CC labeling via iterative max-pool label propagation.

    Each foreground pixel gets a unique initial label (its linear index).
    We then iteratively propagate max labels among 4-connected neighbors
    until convergence. After convergence, connected regions share the same
    (maximum) label.

    Args:
        binary: 2D binary tensor (H, W), float.

    Returns:
        Integer label tensor (H, W), 0 = background.
    """
    H, W = binary.shape
    device = binary.device
    mask = (binary > 0)

    if mask.sum() == 0:
        return torch.zeros(H, W, dtype=torch.long, device=device)

    # Initialize: each foreground pixel gets its linear index + 1 as label
    labels = torch.zeros(H, W, dtype=torch.long, device=device)
    linear_idx = torch.arange(1, H * W + 1, device=device, dtype=torch.long).reshape(H, W)
    labels[mask] = linear_idx[mask]

    # Iterative max-pool propagation (4-connected)
    for _ in range(max(H, W)):
        prev = labels.clone()
        labels_4d = labels.unsqueeze(0).unsqueeze(0).float()
        # Max pool with kernel 3×3, stride 1, pad 1 → 8-connected
        # For 4-connected, we use cross kernel via separate 1D pools
        pooled_h = F.max_pool2d(labels_4d, kernel_size=(3, 1), stride=1, padding=(1, 0))
        pooled_w = F.max_pool2d(labels_4d, kernel_size=(1, 3), stride=1, padding=(0, 1))
        pooled = torch.max(pooled_h, pooled_w).squeeze(0).squeeze(0).long()
        # Only update foreground pixels
        labels[mask] = pooled[mask]
        if torch.equal(labels, prev):
            break

    return labels


def cc_label(binary: torch.Tensor) -> torch.Tensor:
    """Connected-component labeling with automatic backend selection.

    Uses scipy on CPU, GPU-native iterative propagation on CUDA.

    Args:
        binary: 2D binary tensor (H, W).

    Returns:
        Integer label tensor (H, W), 0 = background, >0 = component IDs.
    """
    if binary.device.type == "cpu":
        return _cc_label_scipy(binary)
    return _cc_label_gpu(binary)


# ── Spanning check (true percolation) ─────────────────────────────────────

def _check_spanning(labels: torch.Tensor) -> bool:
    """Check if any component spans edge-to-edge (true percolation).

    A component spans if it touches BOTH top and bottom edges, OR BOTH
    left and right edges of the lattice. This is the classical percolation
    condition (Stauffer & Aharony 1992).

    Args:
        labels: Integer label tensor (H, W), 0 = background.

    Returns:
        True if any component spans the lattice.
    """
    if labels.max() == 0:
        return False

    H, W = labels.shape

    # Get labels touching each edge (exclude background=0)
    top_labels = set(labels[0, :].unique().tolist()) - {0}
    bottom_labels = set(labels[H - 1, :].unique().tolist()) - {0}
    left_labels = set(labels[:, 0].unique().tolist()) - {0}
    right_labels = set(labels[:, W - 1].unique().tolist()) - {0}

    # Vertical spanning: touches top AND bottom
    if top_labels & bottom_labels:
        return True

    # Horizontal spanning: touches left AND right
    if left_labels & right_labels:
        return True

    return False


def _connected_components_largest_ratio(binary: torch.Tensor) -> float:
    """Compute ratio of largest connected component to total foreground.

    Kept for backward compatibility. Uses cc_label() with .long() dtype fix.

    Args:
        binary: 2D binary tensor (H, W).

    Returns:
        Ratio in [0, 1]. Returns 0.0 if no foreground pixels.
    """
    fg_count = binary.sum().item()
    if fg_count < 1:
        return 0.0

    labels = cc_label(binary)
    if labels.max() == 0:
        return 0.0

    # .long() fix: torch.bincount requires int64
    component_sizes = torch.bincount(labels.long().ravel())
    if component_sizes.shape[0] <= 1:
        return 0.0
    largest = component_sizes[1:].max().item()
    return float(largest / fg_count)


def percolation_critical_threshold(
    patch: torch.Tensor,
    threshold_grid: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
) -> float:
    """Find the critical percolation threshold for a grayscale patch.

    Uses the true spanning condition: a threshold τ is considered percolating
    if any connected component at that threshold touches BOTH top-bottom OR
    left-right edges of the lattice (edge-to-edge spanning).

    Sweeps intensity thresholds from high to low. Returns the highest τ
    where a spanning cluster exists.

    Args:
        patch: Grayscale tensor (H, W) with values in [0, 1].
        threshold_grid: Thresholds to sweep (descending order preferred).

    Returns:
        Critical threshold p_c in [0, 1].
        High p_c = dense/connected (spanning cluster at strict threshold).
        Low p_c = sparse/fragmented (spanning only at lenient threshold).
    """
    gray = _to_grayscale(patch).float()

    # Normalize to [0, 1]
    vmin, vmax = gray.min(), gray.max()
    if (vmax - vmin) > 0:
        gray = (gray - vmin) / (vmax - vmin)
    else:
        return 1.0  # uniform → trivially spanning → maximum threshold

    # Sweep from high to low threshold
    sorted_grid = sorted(threshold_grid, reverse=True)
    for tau in sorted_grid:
        binary = (gray >= tau).float()
        labels = cc_label(binary)
        if _check_spanning(labels):
            return float(tau)

    # No threshold produces spanning → return lowest threshold
    return float(sorted_grid[-1]) if sorted_grid else 0.0


def compute_percolation_map(
    image: torch.Tensor,
    window_size: int = 32,
    stride: Optional[int] = None,
    threshold_grid: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    norm_percentiles: Optional[Tuple[float, float]] = None,
) -> torch.Tensor:
    """Compute a pixel-level percolation connectivity map.

    For each local window, finds the critical percolation threshold
    p_c where a spanning cluster first forms (edge-to-edge connectivity).

    Args:
        image: Tensor of shape (C, H, W) or (H, W).
        window_size: Size of sliding window for local percolation analysis.
        stride: Step between windows (default: window_size // 2).
        threshold_grid: Intensity thresholds to sweep per window.
        norm_percentiles: Optional (p1, p99) for dataset-level normalization.
            If provided, raw p_c values are clamped and scaled to [0, 1].

    Returns:
        Tensor of shape (H, W) with float32 p_c values.
        High values = dense/connected regions (spanning at strict threshold).
        Low values = sparse/fragmented regions.
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
                patch, threshold_grid
            )

    # Upsample to original resolution
    perc_map = F.interpolate(
        perc_grid.unsqueeze(0).unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)

    # Apply dataset-level percentile normalization if provided
    if norm_percentiles is not None:
        p_low, p_high = norm_percentiles
        if p_high > p_low:
            perc_map = (perc_map - p_low) / (p_high - p_low)
            perc_map = perc_map.clamp(0.0, 1.0)
        else:
            perc_map = torch.zeros_like(perc_map)

    return perc_map.to(torch.float32)


# ── Dataset-level normalization utilities ─────────────────────────────────

def compute_dataset_percentiles(
    images: list[torch.Tensor],
    window_size: int = 32,
    stride: Optional[int] = None,
    threshold_grid: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> Tuple[float, float]:
    """Compute dataset-level percolation percentiles."""
    all_values = []
    for img in images:
        perc_map = compute_percolation_map(img, window_size, stride, threshold_grid)
        all_values.append(perc_map.flatten())
    all_vals = torch.cat(all_values)
    low_val = float(torch.quantile(all_vals, p_low / 100.0).item())
    high_val = float(torch.quantile(all_vals, p_high / 100.0).item())
    return (low_val, high_val)


def save_percentiles(percentiles: Tuple[float, float], path: str | Path) -> None:
    """Save percentile metadata to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"p1": percentiles[0], "p99": percentiles[1]}, indent=2))


def load_percentiles(path: str | Path) -> Optional[Tuple[float, float]]:
    """Load percentile metadata from JSON file, or None if not found."""
    path = Path(path)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return (float(data["p1"]), float(data["p99"]))
