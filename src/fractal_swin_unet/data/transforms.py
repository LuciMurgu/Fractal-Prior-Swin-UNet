"""Patch-aligned augmentations for image/mask/prior."""

from __future__ import annotations

from typing import Dict, Any

import torch
import torch.nn.functional as F


def random_photometric(
    sample: Dict[str, torch.Tensor],
    generator: torch.Generator,
    config: Dict[str, Any] | None = None,
) -> Dict[str, torch.Tensor]:
    """Apply mild random photometric jitter to image only."""
    if not config or not bool(config.get("enabled", False)):
        return sample
    if "image" not in sample:
        return sample

    out = dict(sample)
    image = out["image"]
    if image.dtype != torch.float32:
        image = image.float()

    brightness = float(config.get("brightness", 0.0))
    contrast = float(config.get("contrast", 0.0))
    gamma = float(config.get("gamma", 0.0))
    clip = bool(config.get("clip", True))

    if brightness > 0:
        delta = (torch.rand((), generator=generator) * 2 - 1) * brightness
        image = image + float(delta.item())

    if contrast > 0:
        factor = 1.0 + (torch.rand((), generator=generator) * 2 - 1) * contrast
        mean = image.mean(dim=(-2, -1), keepdim=True)
        image = (image - mean) * float(factor.item()) + mean

    if gamma > 0:
        g = 1.0 + (torch.rand((), generator=generator) * 2 - 1) * gamma
        image = image.clamp(0.0, 1.0).pow(float(g.item()))

    if clip:
        image = image.clamp(0.0, 1.0)

    out["image"] = image
    return out


def random_flip_rotate(sample: Dict[str, torch.Tensor], generator: torch.Generator) -> Dict[str, torch.Tensor]:
    """Apply the same random flip/rotation to all tensors in sample."""

    keys = [k for k in sample.keys() if k in {"image", "mask", "prior"}]
    if not keys:
        return sample

    do_hflip = torch.rand((), generator=generator).item() > 0.5
    do_vflip = torch.rand((), generator=generator).item() > 0.5
    rot_k = int(torch.randint(0, 4, (1,), generator=generator).item())

    out = dict(sample)
    for key in keys:
        tensor = out[key]
        if do_hflip:
            tensor = torch.flip(tensor, dims=[-1])
        if do_vflip:
            tensor = torch.flip(tensor, dims=[-2])
        if rot_k:
            tensor = torch.rot90(tensor, k=rot_k, dims=[-2, -1])
        out[key] = tensor

    return out


def _generate_elastic_displacement(
    h: int, w: int, alpha: float, sigma: float, generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate smooth random displacement fields for elastic deformation.

    Uses a coarse random field smoothed by bilinear upsampling (fast approximation
    of Gaussian smoothing) to create spatially coherent deformations.

    Args:
        h: Height of the output displacement field.
        w: Width of the output displacement field.
        alpha: Displacement magnitude (in pixels).
        sigma: Controls smoothness — higher means coarser/smoother deformations.
            Implemented as the ratio h/sigma for the coarse grid size.
        generator: Torch random generator for reproducibility.

    Returns:
        Tuple of (dy, dx) displacement fields, each of shape (h, w).
    """
    # Coarse grid size: smaller = smoother deformations
    coarse_h = max(int(h / sigma), 2)
    coarse_w = max(int(w / sigma), 2)

    # Random displacements on coarse grid, then upsample for smoothness
    dy_coarse = (torch.rand(1, 1, coarse_h, coarse_w, generator=generator) * 2 - 1) * alpha
    dx_coarse = (torch.rand(1, 1, coarse_h, coarse_w, generator=generator) * 2 - 1) * alpha

    dy = F.interpolate(dy_coarse, size=(h, w), mode="bilinear", align_corners=False).squeeze()
    dx = F.interpolate(dx_coarse, size=(h, w), mode="bilinear", align_corners=False).squeeze()

    return dy, dx


def _apply_displacement(
    tensor: torch.Tensor, dy: torch.Tensor, dx: torch.Tensor, mode: str = "bilinear",
) -> torch.Tensor:
    """Apply displacement fields to a tensor using grid_sample.

    Args:
        tensor: (C, H, W) tensor to deform.
        dy: (H, W) vertical displacement in pixels.
        dx: (H, W) horizontal displacement in pixels.
        mode: Interpolation mode — 'bilinear' for images/priors, 'nearest' for masks.

    Returns:
        Deformed tensor of same shape.
    """
    c, h, w = tensor.shape

    # Build base grid in [-1, 1] coordinates
    grid_y = torch.linspace(-1, 1, h, device=tensor.device, dtype=tensor.dtype)
    grid_x = torch.linspace(-1, 1, w, device=tensor.device, dtype=tensor.dtype)
    base_grid_y, base_grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")

    # Convert pixel displacements to [-1, 1] normalized space
    norm_dx = dx * 2.0 / max(w - 1, 1)
    norm_dy = dy * 2.0 / max(h - 1, 1)

    grid = torch.stack([base_grid_x + norm_dx, base_grid_y + norm_dy], dim=-1)  # (H, W, 2)
    grid = grid.unsqueeze(0)  # (1, H, W, 2)

    out = F.grid_sample(
        tensor.unsqueeze(0), grid, mode=mode, padding_mode="reflection", align_corners=True,
    )
    return out.squeeze(0)


def random_elastic(
    sample: Dict[str, torch.Tensor],
    generator: torch.Generator,
    config: Dict[str, Any] | None = None,
) -> Dict[str, torch.Tensor]:
    """Apply random elastic deformation to image, mask, and prior consistently.

    Config keys:
        enabled: bool (default False)
        alpha: float — displacement magnitude in pixels (default 20.0)
        sigma: float — smoothness factor (default 6.0)
        p: float — probability of applying (default 0.5)
    """
    if not config or not bool(config.get("enabled", False)):
        return sample
    if "image" not in sample:
        return sample

    p = float(config.get("p", 0.5))
    if torch.rand((), generator=generator).item() > p:
        return sample

    alpha = float(config.get("alpha", 20.0))
    sigma = float(config.get("sigma", 6.0))

    _, h, w = sample["image"].shape
    dy, dx = _generate_elastic_displacement(h, w, alpha, sigma, generator)

    out = dict(sample)
    for key in ["image", "prior"]:
        if key in out:
            out[key] = _apply_displacement(out[key], dy, dx, mode="bilinear")
    if "mask" in out:
        out["mask"] = _apply_displacement(out["mask"], dy, dx, mode="nearest")

    return out


def random_scale(
    sample: Dict[str, torch.Tensor],
    generator: torch.Generator,
    config: Dict[str, Any] | None = None,
) -> Dict[str, torch.Tensor]:
    """Apply random scale augmentation then center-crop back to original size.

    Config keys:
        enabled: bool (default False)
        scale_range: [min, max] scale factors (default [0.8, 1.2])
        p: float — probability of applying (default 0.5)
    """
    if not config or not bool(config.get("enabled", False)):
        return sample
    if "image" not in sample:
        return sample

    p = float(config.get("p", 0.5))
    if torch.rand((), generator=generator).item() > p:
        return sample

    scale_range = config.get("scale_range", [0.8, 1.2])
    lo, hi = float(scale_range[0]), float(scale_range[1])
    scale = lo + (hi - lo) * torch.rand((), generator=generator).item()

    _, h, w = sample["image"].shape
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))

    out = dict(sample)
    for key in ["image", "prior"]:
        if key in out:
            t = out[key].unsqueeze(0)  # (1, C, H, W)
            t = F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)
            out[key] = t.squeeze(0)

    if "mask" in out:
        t = out["mask"].unsqueeze(0)
        t = F.interpolate(t, size=(new_h, new_w), mode="nearest")
        out["mask"] = t.squeeze(0)

    # Center crop or pad back to original size
    for key in ["image", "mask", "prior"]:
        if key not in out:
            continue
        tensor = out[key]
        _, th, tw = tensor.shape

        if th >= h and tw >= w:
            # Center crop
            top = (th - h) // 2
            left = (tw - w) // 2
            out[key] = tensor[:, top : top + h, left : left + w]
        else:
            # Pad with reflection
            pad_top = max((h - th) // 2, 0)
            pad_left = max((w - tw) // 2, 0)
            pad_bottom = h - th - pad_top
            pad_right = w - tw - pad_left
            out[key] = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")

    return out


def random_gaussian_noise(
    sample: Dict[str, torch.Tensor],
    generator: torch.Generator,
    config: Dict[str, Any] | None = None,
) -> Dict[str, torch.Tensor]:
    """Add random Gaussian noise to the image only.

    Config keys:
        enabled: bool (default False)
        std: float — noise standard deviation (default 0.02)
        p: float — probability of applying (default 0.5)
    """
    if not config or not bool(config.get("enabled", False)):
        return sample
    if "image" not in sample:
        return sample

    p = float(config.get("p", 0.5))
    if torch.rand((), generator=generator).item() > p:
        return sample

    std = float(config.get("std", 0.02))
    out = dict(sample)
    noise = torch.randn_like(out["image"]) * std
    out["image"] = (out["image"] + noise).clamp(0.0, 1.0)
    return out
