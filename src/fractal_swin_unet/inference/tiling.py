"""Tiled inference utilities with seam-free blending."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def hann2d(ph: int, pw: int, device: torch.device, dtype: torch.dtype, eps: float) -> torch.Tensor:
    win_h = torch.hann_window(ph, periodic=False, device=device, dtype=dtype)
    win_w = torch.hann_window(pw, periodic=False, device=device, dtype=dtype)
    weight = win_h[:, None] * win_w[None, :]
    weight = torch.clamp(weight, min=eps)
    return weight


def _normalize_image(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 3:
        return image.unsqueeze(0)
    return image


def _get_model_prior_channels(model: torch.nn.Module) -> int:
    """Get the number of prior channels the model expects."""
    if hasattr(model, 'prior_channels'):
        return model.prior_channels
    return 1


def _model_needs_lfd(model: torch.nn.Module) -> bool:
    """Check if the model is a FractalPriorSwinUNet that needs an LFD map."""
    cls_name = type(model).__name__
    return cls_name == "FractalPriorSwinUNet"


def _build_tile_prior_provider(
    model: torch.nn.Module,
    fractal_prior_config: Optional[Dict[str, Any]] = None,
) -> Optional["FractalPriorProvider"]:  # noqa: F821
    """Build a FractalPriorProvider that matches what training used.

    If fractal_prior_config is provided (from the YAML), use it directly.
    Otherwise fall back to heuristic based on model.prior_channels.
    """
    from ..fractal.provider import FractalPriorConfig, FractalPriorProvider
    from ..data import _fractal_config_from_dict

    if fractal_prior_config is not None:
        cfg = _fractal_config_from_dict(fractal_prior_config)
        return FractalPriorProvider(cfg)

    # Legacy fallback: guess from model.prior_channels
    prior_channels = _get_model_prior_channels(model)
    if prior_channels > 1:
        cfg = FractalPriorConfig(
            enabled=True,
            multi_scale_enabled=True,
            multi_scale_factors=(1, 2, 4),
            include_lacunarity=(prior_channels >= 4),
        )
    else:
        cfg = FractalPriorConfig(enabled=True)
    return FractalPriorProvider(cfg)


def _compute_tile_prior(
    provider: "FractalPriorProvider",  # noqa: F821
    tile: torch.Tensor,
) -> torch.Tensor:
    """Compute prior channels for a single tile using the provider.

    Args:
        provider: Configured FractalPriorProvider.
        tile: (C, H, W) image tile.

    Returns:
        (prior_channels, H, W) tensor.
    """
    prior = provider.get_patch("tile", image_patch=tile)
    if prior.ndim == 2:
        prior = prior.unsqueeze(0)
    return prior


def tiled_predict_proba(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    patch_size: Tuple[int, int],
    stride: Tuple[int, int],
    blend: str = "hann",
    pad_mode: str = "reflect",
    batch_tiles: int = 8,
    device: torch.device | str | None = None,
    eps: float = 1e-6,
    fractal_prior_config: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    model.eval()
    if device is None:
        device = image_tensor.device
    device = torch.device(device)

    image = _normalize_image(image_tensor).to(device)
    if image.ndim != 4:
        raise ValueError("image_tensor must be (C,H,W) or (1,C,H,W)")

    _, channels, height, width = image.shape
    ph, pw = patch_size
    sh, sw = stride

    # --- Build prior provider matching the training config ---
    needs_lfd = _model_needs_lfd(model)
    prior_provider = None
    if needs_lfd:
        prior_provider = _build_tile_prior_provider(model, fractal_prior_config)

    pad_h = max(ph - height, 0)
    pad_w = max(pw - width, 0)
    if pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode=pad_mode)

    _, _, height_pad, width_pad = image.shape

    ys = list(range(0, max(height_pad - ph, 0) + 1, sh))
    xs = list(range(0, max(width_pad - pw, 0) + 1, sw))
    if ys[-1] != height_pad - ph:
        ys.append(height_pad - ph)
    if xs[-1] != width_pad - pw:
        xs.append(width_pad - pw)

    if blend == "hann":
        weight = hann2d(ph, pw, device=device, dtype=image.dtype, eps=eps)
    else:
        weight = torch.ones((ph, pw), device=device, dtype=image.dtype)

    prob_sum = torch.zeros((1, 1, height_pad, width_pad), device=device, dtype=image.dtype)
    w_sum = torch.zeros((1, 1, height_pad, width_pad), device=device, dtype=image.dtype)

    tiles = []
    lfd_tiles = []
    coords = []

    def _flush_tiles() -> None:
        if not tiles:
            return
        batch = torch.stack(tiles, dim=0)
        with torch.no_grad():
            if needs_lfd and lfd_tiles:
                lfd_batch = torch.stack(lfd_tiles, dim=0).to(device)
                out = model(batch, lfd_map=lfd_batch)
            else:
                out = model(batch)
            logits = out["logits"] if isinstance(out, dict) else out
            probs = torch.sigmoid(logits)
        for idx, (y, x) in enumerate(coords):
            prob_sum[:, :, y : y + ph, x : x + pw] += probs[idx : idx + 1] * weight
            w_sum[:, :, y : y + ph, x : x + pw] += weight
        tiles.clear()
        lfd_tiles.clear()
        coords.clear()

    for y in ys:
        for x in xs:
            tile = image[:, :, y : y + ph, x : x + pw].squeeze(0)
            tiles.append(tile)
            coords.append((y, x))
            if prior_provider is not None:
                # Compute prior on CPU to avoid VRAM pressure
                tile_cpu = tile.cpu()
                prior_tile = _compute_tile_prior(prior_provider, tile_cpu)
                lfd_tiles.append(prior_tile)
            if len(tiles) >= batch_tiles:
                _flush_tiles()

    _flush_tiles()

    probs = prob_sum / (w_sum + eps)
    probs = probs[:, :, :height, :width]
    return probs

