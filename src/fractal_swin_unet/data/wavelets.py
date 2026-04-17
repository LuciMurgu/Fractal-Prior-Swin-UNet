"""Deterministic wavelet feature projections for retinal images."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F


_DEFAULT_BANDS = ("ll", "lh", "hl", "hh")
_DEFAULT_PROJECTIONS = ("gray",)


def wavelet_channel_count(base_channels: int, wavelet_cfg: dict[str, Any] | None) -> int:
    """Return effective input channels after optional wavelet projection stacking."""
    if not wavelet_cfg or not bool(wavelet_cfg.get("enabled", False)):
        return int(base_channels)
    levels = max(1, int(wavelet_cfg.get("levels", 1)))
    bands = _normalize_bands(wavelet_cfg.get("bands", list(_DEFAULT_BANDS)))
    projections = _normalize_projections(wavelet_cfg.get("projections", list(_DEFAULT_PROJECTIONS)))
    include_input = bool(wavelet_cfg.get("include_input", True))
    projected = len(projections) * levels * len(bands)
    return int(base_channels if include_input else 0) + projected


def apply_wavelet_projections(image: torch.Tensor, wavelet_cfg: dict[str, Any] | None) -> torch.Tensor:
    """Append deterministic wavelet channels to the image tensor based on config."""
    if image.ndim != 3:
        raise ValueError("image must have shape (C, H, W)")
    if not wavelet_cfg or not bool(wavelet_cfg.get("enabled", False)):
        return image

    levels = max(1, int(wavelet_cfg.get("levels", 1)))
    bands = _normalize_bands(wavelet_cfg.get("bands", list(_DEFAULT_BANDS)))
    projections = _normalize_projections(wavelet_cfg.get("projections", list(_DEFAULT_PROJECTIONS)))
    include_input = bool(wavelet_cfg.get("include_input", True))
    normalize = bool(wavelet_cfg.get("normalize_per_channel", True))
    eps = float(wavelet_cfg.get("eps", 1e-6))
    mode = str(wavelet_cfg.get("upsample_mode", "bilinear"))

    _, height, width = image.shape
    features: list[torch.Tensor] = []
    for projection_name in projections:
        base = _projection_channel(image, projection_name)
        current = base
        for _ in range(levels):
            ll, lh, hl, hh = _haar_dwt2d(current)
            bands_map = {"ll": ll, "lh": lh, "hl": hl, "hh": hh}
            for band_name in bands:
                up = F.interpolate(
                    bands_map[band_name].unsqueeze(0).unsqueeze(0),
                    size=(height, width),
                    mode=mode,
                    align_corners=False if mode in {"bilinear", "bicubic"} else None,
                ).squeeze(0)
                if normalize:
                    up = _normalize01(up, eps=eps)
                features.append(up)
            current = ll

    wavelet_stack = torch.cat(features, dim=0)
    if include_input:
        return torch.cat([image, wavelet_stack], dim=0)
    return wavelet_stack


def _normalize_bands(bands: Sequence[str] | None) -> list[str]:
    values = [str(b).lower() for b in (bands or _DEFAULT_BANDS)]
    if not values:
        values = list(_DEFAULT_BANDS)
    allowed = {"ll", "lh", "hl", "hh"}
    invalid = [b for b in values if b not in allowed]
    if invalid:
        raise ValueError(f"Unsupported wavelet bands: {invalid}")
    return values


def _normalize_projections(projections: Iterable[str] | None) -> list[str]:
    values = [str(p).lower() for p in (projections or _DEFAULT_PROJECTIONS)]
    return values or list(_DEFAULT_PROJECTIONS)


def _projection_channel(image: torch.Tensor, name: str) -> torch.Tensor:
    if image.shape[0] == 1:
        channel0 = image[0]
        if name in {"gray", "green", "red", "blue", "mean", "max", "channel0"}:
            return channel0
    if name == "gray":
        if image.shape[0] >= 3:
            return 0.2989 * image[0] + 0.5870 * image[1] + 0.1140 * image[2]
        return image.mean(dim=0)
    if name == "green":
        return image[1] if image.shape[0] > 1 else image[0]
    if name == "red":
        return image[0]
    if name == "blue":
        return image[2] if image.shape[0] > 2 else image[0]
    if name == "mean":
        return image.mean(dim=0)
    if name == "max":
        return image.max(dim=0).values
    if name.startswith("channel"):
        suffix = name.replace("channel_", "").replace("channel", "")
        idx = int(suffix)
        idx = max(0, min(idx, image.shape[0] - 1))
        return image[idx]
    raise ValueError(f"Unsupported wavelet projection '{name}'")


def _haar_dwt2d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.ndim != 2:
        raise ValueError("x must have shape (H, W)")
    x = _pad_to_even(x)
    a = x[0::2, 0::2]
    b = x[0::2, 1::2]
    c = x[1::2, 0::2]
    d = x[1::2, 1::2]
    ll = (a + b + c + d) * 0.25
    lh = (a - b + c - d) * 0.25
    hl = (a + b - c - d) * 0.25
    hh = (a - b - c + d) * 0.25
    return ll, lh, hl, hh


def _pad_to_even(x: torch.Tensor) -> torch.Tensor:
    h, w = x.shape
    pad_h = h % 2
    pad_w = w % 2
    if pad_h:
        x = torch.cat([x, x[-1:, :]], dim=0)
    if pad_w:
        x = torch.cat([x, x[:, -1:]], dim=1)
    return x


def _normalize01(x: torch.Tensor, eps: float) -> torch.Tensor:
    x_min = x.amin()
    x_max = x.amax()
    return (x - x_min) / (x_max - x_min + eps)

