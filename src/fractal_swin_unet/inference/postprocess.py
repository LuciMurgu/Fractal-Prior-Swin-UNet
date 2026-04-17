"""Morphological post-processing for binary vessel segmentation masks."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F


def _disk_kernel(radius: int, device: torch.device) -> torch.Tensor:
    """Create a circular structuring element."""
    size = 2 * radius + 1
    y, x = torch.meshgrid(
        torch.arange(size, device=device, dtype=torch.float32) - radius,
        torch.arange(size, device=device, dtype=torch.float32) - radius,
        indexing="ij",
    )
    kernel = ((x ** 2 + y ** 2) <= radius ** 2).float()
    return kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)


def _morphological_close(mask: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Morphological closing: dilate then erode."""
    pad = kernel.shape[-1] // 2
    # Dilate
    dilated = F.conv2d(mask, kernel, padding=pad)
    dilated = (dilated > 0).float()
    # Erode
    k_sum = kernel.sum()
    eroded = F.conv2d(dilated, kernel, padding=pad)
    eroded = (eroded >= k_sum).float()
    return eroded


def _remove_small_components(
    mask: torch.Tensor, min_size: int = 15,
) -> torch.Tensor:
    """Remove connected components smaller than min_size pixels.
    
    Uses a simple flood-fill approach. For GPU tensors, falls back to
    scipy on CPU then transfers back.
    """
    device = mask.device
    result = mask.clone()

    for b in range(mask.shape[0]):
        binary = mask[b, 0].cpu().numpy() > 0.5
        try:
            from scipy import ndimage
            labeled, num_features = ndimage.label(binary)
            if num_features == 0:
                continue
            sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
            small = [i + 1 for i, s in enumerate(sizes) if s < min_size]
            if small:
                remove_mask = torch.zeros_like(mask[b, 0], device="cpu")
                for label_id in small:
                    remove_mask[labeled == label_id] = 1.0
                result[b, 0] = result[b, 0] - remove_mask.to(device)
                result = result.clamp(0, 1)
        except ImportError:
            # Without scipy, skip small component removal
            pass

    return result


def postprocess_vessel_mask(
    probs: torch.Tensor,
    threshold: float = 0.5,
    config: Dict[str, Any] | None = None,
) -> torch.Tensor:
    """Apply morphological post-processing to vessel probability map.

    Args:
        probs: Probability tensor of shape (B, 1, H, W) in [0, 1].
        threshold: Binarization threshold.
        config: Post-processing configuration dict:
            - enabled: bool (default True)
            - min_component_size: int (default 15) — remove components smaller than this
            - closing_radius: int (default 1) — disk radius for morphological closing
            - fill_holes: bool (default False) — fill small holes in vessels

    Returns:
        Binary mask tensor of shape (B, 1, H, W) with values 0.0 or 1.0.
    """
    if config is None:
        config = {}

    if not bool(config.get("enabled", True)):
        return (probs >= threshold).float()

    min_size = int(config.get("min_component_size", 15))
    closing_radius = int(config.get("closing_radius", 1))

    # Binarize
    mask = (probs >= threshold).float()

    # Morphological closing to fill 1-px gaps
    if closing_radius > 0:
        kernel = _disk_kernel(closing_radius, device=mask.device)
        mask = _morphological_close(mask, kernel)

    # Remove small connected components (noise)
    if min_size > 0:
        mask = _remove_small_components(mask, min_size=min_size)

    return mask
