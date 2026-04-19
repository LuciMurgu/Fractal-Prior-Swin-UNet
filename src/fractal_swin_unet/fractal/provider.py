"""Fractal prior provider with caching and precompute support."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .cache_disk import cache_path, load_cached_map, make_cache_key, metadata_path, save_cached_map, save_metadata
from .engine import DbcLfdConfig, DbcLfdEngine


def _normalize_grayscale(image: torch.Tensor, mode: str) -> torch.Tensor:
    if image.ndim == 3:
        if mode == "luminance":
            return image.mean(dim=0)
        return image.mean(dim=0)
    return image


def _normalize_map(map_tensor: torch.Tensor, mode: str, eps: float) -> torch.Tensor:
    if mode == "per_image":
        min_val = map_tensor.min()
        max_val = map_tensor.max()
        return (map_tensor - min_val) / (max_val - min_val + eps)
    return map_tensor


@dataclass
class FractalPriorConfig:
    enabled: bool = True
    engine: str = "dbc_lfd"
    grayscale_mode: str = "luminance"
    normalize_mode: str = "per_patch"
    normalize_eps: float = 1e-6
    lfd_window: int = 15
    lfd_stride: int = 4
    lfd_box_sizes: Tuple[int, ...] = (2, 4, 8, 16)
    lfd_fast_mode: bool = True
    caching_enabled: bool = False
    caching_mode: str = "off"
    cache_dir: str = "cache/fractal_prior"
    cache_write: bool = True
    cache_read: bool = True
    precompute_enabled: bool = False
    precompute_store_full: bool = False
    precompute_dtype: str = "float16"
    multi_scale_enabled: bool = False
    multi_scale_factors: Tuple[int, ...] = (1,)
    include_lacunarity: bool = False
    include_percolation: bool = False
    # PDE-based enhancement channels
    pde_channels_enabled: bool = False
    pde_anisotropic_diffusion: bool = True
    pde_frangi: bool = True
    pde_meijering: bool = True
    pde_fractional_laplacian: bool = True
    pde_diffusion_kappa: float = 30.0
    pde_diffusion_n_iter: int = 15
    pde_frangi_sigmas: Tuple[float, ...] = (1.0, 2.0, 4.0)
    pde_meijering_sigmas: Tuple[float, ...] = (0.5, 1.0, 2.0)
    pde_frac_alpha: float = 0.5
    pde_only: bool = False  # If True, skip LFD and return only PDE channels
    pde_cache_dir: str = ""  # If set, load precomputed PDE maps from this dir


class FractalPriorProvider:
    def __init__(self, config: FractalPriorConfig) -> None:
        self.config = config
        self.engine = DbcLfdEngine(
            DbcLfdConfig(
                window=config.lfd_window,
                stride=config.lfd_stride,
                box_sizes=config.lfd_box_sizes,
                fast_mode=config.lfd_fast_mode,
            )
        )
        self.memory_cache: Dict[str, torch.Tensor] = {}
        # PDE cache: maps sample_id -> (4, H, W) tensor of precomputed PDE maps
        self._pde_cache: Dict[str, torch.Tensor] = {}
        self._pde_cache_dir: Optional[Path] = None
        if config.pde_cache_dir:
            self._pde_cache_dir = Path(config.pde_cache_dir)
            if self._pde_cache_dir.exists():
                import logging
                logging.getLogger(__name__).info(
                    f"PDE cache dir: {self._pde_cache_dir} ({len(list(self._pde_cache_dir.glob('*.pt')))} cached)"
                )

    def _cache_key(self, sample_id: str) -> str:
        params = {
            "engine": self.engine.name,
            "lfd.window": self.config.lfd_window,
            "lfd.stride": self.config.lfd_stride,
            "lfd.box_sizes": list(self.config.lfd_box_sizes),
            "lfd.fast_mode": self.config.lfd_fast_mode,
            "normalize.mode": self.config.normalize_mode,
            "grayscale.mode": self.config.grayscale_mode,
        }
        return make_cache_key(sample_id, self.engine.name, params)

    def get_full(self, sample_id: str, image_full: torch.Tensor) -> torch.Tensor:
        """Compute full-image prior with all channels (LFD + lacunarity + percolation).

        Returns (C, H, W) tensor, cached in memory or disk.
        """
        if not self.config.enabled:
            raise ValueError("Fractal prior disabled.")

        image_gray = _normalize_grayscale(image_full, self.config.grayscale_mode).float()
        cache_key = self._cache_key(sample_id)

        if self.config.caching_enabled and self.config.caching_mode == "memory":
            if cache_key in self.memory_cache:
                return self.memory_cache[cache_key]

        if self.config.caching_enabled and self.config.caching_mode == "disk" and self.config.cache_read:
            path = cache_path(self.config.cache_dir, cache_key)
            if path.exists():
                return load_cached_map(path)

        prior_full = self.engine.compute_full(image_gray)
        prior_full = _normalize_map(prior_full, self.config.normalize_mode, self.config.normalize_eps)

        # Ensure 3D
        if prior_full.ndim == 2:
            prior_full = prior_full.unsqueeze(0)

        # Append lacunarity channel to full-image prior (so patches get it for free)
        if self.config.include_lacunarity:
            from .lacunarity import compute_lacunarity_map
            height, width = prior_full.shape[-2], prior_full.shape[-1]
            lac_window = max(32, min(height, width) // 3)
            lac = compute_lacunarity_map(
                image_full,
                window_size=min(lac_window, min(height, width)),
                box_sizes=(2, 4, 8, 16),
            )
            prior_full = torch.cat([prior_full, lac.unsqueeze(0)], dim=0)

        # Append percolation channel
        if self.config.include_percolation:
            from .percolation import compute_percolation_map
            height, width = prior_full.shape[-2], prior_full.shape[-1]
            perc_grid = tuple(round(0.05 + i * 0.05, 2) for i in range(19))
            perc_window = max(32, min(height, width) // 3)
            perc = compute_percolation_map(
                image_full,
                window_size=min(perc_window, min(height, width)),
                threshold_grid=perc_grid,
            )
            prior_full = torch.cat([prior_full, perc.unsqueeze(0)], dim=0)

        if self.config.caching_enabled and self.config.caching_mode == "memory":
            if self.config.precompute_store_full:
                self.memory_cache[cache_key] = prior_full

        if self.config.caching_enabled and self.config.caching_mode == "disk" and self.config.cache_write:
            path = cache_path(self.config.cache_dir, cache_key)
            meta_path = metadata_path(self.config.cache_dir, cache_key)
            save_cached_map(path, prior_full, dtype=self.config.precompute_dtype)
            save_metadata(meta_path, {"sample_id": sample_id, "params": self.engine.params_signature()})

        return prior_full

    def _load_pde_cached(self, sample_id: str) -> Optional[torch.Tensor]:
        """Try to load precomputed PDE maps from disk cache.

        Returns (4, H, W) tensor or None if not cached.
        """
        if sample_id in self._pde_cache:
            return self._pde_cache[sample_id]
        if self._pde_cache_dir is None:
            return None
        cache_path = self._pde_cache_dir / f"{sample_id}.pt"
        if not cache_path.exists():
            return None
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        pde_maps = data["pde_maps"].float()  # (4, H, W)
        self._pde_cache[sample_id] = pde_maps
        return pde_maps

    def _get_pde_patch(
        self,
        sample_id: str,
        image_patch: torch.Tensor,
        patch_box: Tuple[int, int, int, int] | None = None,
    ) -> torch.Tensor:
        """Get PDE channels for a patch, using cache if available."""
        cached = self._load_pde_cached(sample_id)
        if cached is not None and patch_box is not None:
            top, left, height, width = patch_box
            return cached[:, top : top + height, left : left + width]  # (4, pH, pW)
        # Fallback: compute on the fly
        return _compute_pde_channels(image_patch, self.config)

    def get_patch(
        self,
        sample_id: str,
        image_patch: torch.Tensor,
        image_full: torch.Tensor | None = None,
        patch_box: Tuple[int, int, int, int] | None = None,
    ) -> torch.Tensor:
        if not self.config.enabled:
            raise ValueError("Fractal prior disabled.")

        if self.config.precompute_enabled and image_full is not None and patch_box is not None:
            # Full-image path: get_full already includes LFD + lacunarity + percolation
            prior_full = self.get_full(sample_id, image_full)
            top, left, height, width = patch_box
            result = prior_full[:, top : top + height, left : left + width]
        else:
            image_gray = _normalize_grayscale(image_patch, self.config.grayscale_mode).float()
            prior_patch = self.engine.compute_patch(image_gray)
            prior_patch = _normalize_map(prior_patch, self.config.normalize_mode, self.config.normalize_eps)
            prior_patch = prior_patch.float()

            # PDE-only mode: skip LFD, return only PDE channels
            if self.config.pde_channels_enabled and self.config.pde_only:
                return self._get_pde_patch(sample_id, image_patch, patch_box)

            if self.config.multi_scale_enabled:
                result = _build_multi_scale_stack(
                    prior_patch,
                    factors=self.config.multi_scale_factors,
                    eps=self.config.normalize_eps,
                    include_lacunarity=self.config.include_lacunarity,
                    image_patch=image_patch,
                )
            else:
                result = prior_patch

            # Ensure 3D (C, H, W) for channel concatenation
            if result.ndim == 2:
                result = result.unsqueeze(0)  # (1, H, W)

            # Append lacunarity channel (only when NOT using precompute path)
            if self.config.include_lacunarity and not self.config.multi_scale_enabled:
                from .lacunarity import compute_lacunarity_map
                height, width = result.shape[-2], result.shape[-1]
                lac_window = max(32, min(height, width) // 3)
                lac = compute_lacunarity_map(
                    image_patch,
                    window_size=min(lac_window, min(height, width)),
                    box_sizes=(2, 4, 8, 16),
                )
                result = torch.cat([result, lac.unsqueeze(0)], dim=0)

            if self.config.include_percolation:
                from .percolation import compute_percolation_map
                height, width = result.shape[-2], result.shape[-1]
                perc_grid = tuple(round(0.05 + i * 0.05, 2) for i in range(19))
                perc_window = max(32, min(height, width) // 3)
                perc = compute_percolation_map(
                    image_patch,
                    window_size=min(perc_window, min(height, width)),
                    threshold_grid=perc_grid,
                )
                result = torch.cat([result, perc.unsqueeze(0)], dim=0)

        # Append PDE channels if enabled
        if self.config.pde_channels_enabled:
            pde_maps = self._get_pde_patch(sample_id, image_patch, patch_box)
            result = torch.cat([result, pde_maps], dim=0)  # (C+N_pde, H, W)

        return result


def _build_multi_scale_stack(
    prior_patch: torch.Tensor,
    factors: Tuple[int, ...],
    eps: float,
    include_lacunarity: bool = False,
    image_patch: torch.Tensor | None = None,
) -> torch.Tensor:
    if prior_patch.ndim != 2:
        raise ValueError("prior_patch must be 2D (H, W).")
    height, width = prior_patch.shape
    channels = []
    unique_factors = sorted({max(1, int(f)) for f in factors}) or [1]
    for factor in unique_factors:
        if factor == 1:
            scaled = prior_patch
        else:
            pooled = F.avg_pool2d(
                prior_patch.unsqueeze(0).unsqueeze(0),
                kernel_size=factor,
                stride=factor,
                ceil_mode=True,
            )
            scaled = F.interpolate(pooled, size=(height, width), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
        scaled = _normalize_map(scaled, mode="per_image", eps=eps)
        channels.append(scaled)

    if include_lacunarity and image_patch is not None:
        from .lacunarity import compute_lacunarity_map
        lac_window = max(32, min(height, width) // 3)
        lac = compute_lacunarity_map(
            image_patch,
            window_size=min(lac_window, min(height, width)),
            box_sizes=(2, 4, 8, 16),
        )
        channels.append(lac)

    return torch.stack(channels, dim=0)


def _compute_pde_channels(
    image_patch: torch.Tensor,
    config: FractalPriorConfig,
) -> torch.Tensor:
    """Compute PDE-based enhancement channels for a patch.

    Args:
        image_patch: (C, H, W) image tensor.
        config: FractalPriorConfig with PDE settings.

    Returns:
        (N_pde, H, W) tensor of PDE feature channels.
    """
    from ..pde import (
        perona_malik_diffusion,
        frangi_vesselness,
        meijering_neuriteness,
        fractional_laplacian,
    )

    channels = []

    if config.pde_anisotropic_diffusion:
        diff = perona_malik_diffusion(
            image_patch,
            kappa=config.pde_diffusion_kappa,
            n_iter=config.pde_diffusion_n_iter,
        )
        if diff.ndim == 3:
            diff = diff[0]  # (H, W)
        channels.append(diff)

    if config.pde_frangi:
        frangi = frangi_vesselness(
            image_patch,
            sigmas=config.pde_frangi_sigmas,
        )
        if frangi.ndim == 3:
            frangi = frangi[0]
        channels.append(frangi)

    if config.pde_meijering:
        meij = meijering_neuriteness(
            image_patch,
            sigmas=config.pde_meijering_sigmas,
        )
        if meij.ndim == 3:
            meij = meij[0]
        channels.append(meij)

    if config.pde_fractional_laplacian:
        frac = fractional_laplacian(
            image_patch,
            alpha=config.pde_frac_alpha,
        )
        if frac.ndim == 3:
            frac = frac[0]
        channels.append(frac)

    return torch.stack(channels, dim=0)  # (N_pde, H, W)
