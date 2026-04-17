"""Dataset utilities for random patch sampling."""

from __future__ import annotations

from typing import Iterator, Tuple, Dict, Any

import torch
from torch.utils.data import IterableDataset, get_worker_info

from .sampling import PatchCenterSampler, PatchSamplingConfig
from .transforms import random_flip_rotate, random_photometric, random_elastic, random_scale, random_gaussian_noise
from ..fractal.provider import FractalPriorConfig, FractalPriorProvider


def _to_patch_size(patch_size: int | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(patch_size, int):
        return (patch_size, patch_size)
    return patch_size


def _sampling_config_from_dict(config: Dict[str, Any] | None) -> PatchSamplingConfig:
    if not config:
        return PatchSamplingConfig()
    return PatchSamplingConfig(
        enabled=bool(config.get("enabled", False)),
        mode=str(config.get("mode", "uniform")),
        p_vessel=float(config.get("p_vessel", 0.7)),
        p_background=float(config.get("p_background", 0.3)),
        vessel_buffer=int(config.get("vessel_buffer", 3)),
        background_buffer=int(config.get("background_buffer", 0)),
        min_vessel_fraction_in_patch=float(config.get("min_vessel_fraction_in_patch", 0.0)),
        max_retries=int(config.get("max_retries", 30)),
        seed=config.get("seed"),
    )


def _fractal_config_from_dict(config: Dict[str, Any] | None) -> FractalPriorConfig:
    if not config:
        return FractalPriorConfig(enabled=False)
    caching = config.get("caching", {})
    lfd = config.get("lfd", {})
    grayscale = config.get("grayscale", {})
    normalize = config.get("normalize", {})
    precompute = config.get("precompute", {})
    multi_scale = config.get("multi_scale", {})
    return FractalPriorConfig(
        enabled=bool(config.get("enabled", True)),
        engine=str(config.get("engine", "dbc_lfd")),
        grayscale_mode=str(grayscale.get("mode", "luminance")),
        normalize_mode=str(normalize.get("mode", "per_patch")),
        normalize_eps=float(normalize.get("eps", 1e-6)),
        lfd_window=int(lfd.get("window", 15)),
        lfd_stride=int(lfd.get("stride", 4)),
        lfd_box_sizes=tuple(lfd.get("box_sizes", [2, 4, 8, 16])),
        lfd_fast_mode=bool(lfd.get("fast_mode", True)),
        caching_enabled=bool(caching.get("enabled", False)),
        caching_mode=str(caching.get("mode", "off")),
        cache_dir=str(caching.get("cache_dir", "cache/fractal_prior")),
        cache_write=bool(caching.get("write", True)),
        cache_read=bool(caching.get("read", True)),
        precompute_enabled=bool(precompute.get("enabled", False)),
        precompute_store_full=bool(precompute.get("store_full_map_in_memory", False)),
        precompute_dtype=str(precompute.get("dtype", "float16")),
        multi_scale_enabled=bool(multi_scale.get("enabled", False)),
        multi_scale_factors=tuple(multi_scale.get("factors", [1])),
        include_lacunarity=bool(multi_scale.get("include_lacunarity", False)) or bool(config.get("include_lacunarity", False)),
        include_percolation=bool(config.get("include_percolation", False)),
        # PDE channels
        pde_channels_enabled=bool(config.get("pde_channels_enabled", False)),
        pde_anisotropic_diffusion=bool(config.get("pde_anisotropic_diffusion", True)),
        pde_frangi=bool(config.get("pde_frangi", True)),
        pde_meijering=bool(config.get("pde_meijering", True)),
        pde_fractional_laplacian=bool(config.get("pde_fractional_laplacian", True)),
        pde_diffusion_kappa=float(config.get("pde_diffusion_kappa", 30.0)),
        pde_diffusion_n_iter=int(config.get("pde_diffusion_n_iter", 15)),
        pde_frangi_sigmas=tuple(config.get("pde_frangi_sigmas", [1.0, 2.0, 4.0])),
        pde_meijering_sigmas=tuple(config.get("pde_meijering_sigmas", [0.5, 1.0, 2.0])),
        pde_frac_alpha=float(config.get("pde_frac_alpha", 0.5)),
        pde_only=bool(config.get("pde_only", False)),
        pde_cache_dir=str(config.get("pde_cache_dir", "")),
    )


class InfiniteRandomPatchDataset(IterableDataset):
    """Infinite random patch dataset for a single image and mask."""

    def __init__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor | None = None,
        patch_size: int | Tuple[int, int] = 128,
        seed: int = 0,
        sampling_config: Dict[str, Any] | None = None,
        sample_id: str | None = None,
        fractal_prior_config: Dict[str, Any] | None = None,
        return_fractal_prior: bool = False,
        photometric_config: Dict[str, Any] | None = None,
    ) -> None:
        """Create an infinite random patch dataset."""

        if image.ndim != 3:
            raise ValueError("Image must have shape (C, H, W).")

        self.image = image
        self.mask = mask
        self.patch_h, self.patch_w = _to_patch_size(patch_size)
        self.seed = seed
        self.sample_id = sample_id or "default"

        self.sampling_config = _sampling_config_from_dict(sampling_config)
        self.sampler = PatchCenterSampler((self.patch_h, self.patch_w), self.sampling_config)
        self._mask_cache: Dict[str, Dict[str, torch.Tensor]] = {}

        self.return_fractal_prior = return_fractal_prior
        self.fractal_config = _fractal_config_from_dict(fractal_prior_config)
        self.fractal_provider = FractalPriorProvider(self.fractal_config) if self.fractal_config.enabled else None
        self.photometric_config = photometric_config or {}
        raw_sampling = sampling_config or {}
        self.fractal_hard_cfg = raw_sampling.get("fractal_hard", {})
        self.spatial_aug_config = (photometric_config or {}).get("spatial", {})

        _, height, width = image.shape
        if self.patch_h > height or self.patch_w > width:
            raise ValueError("Patch size must fit within the image.")

    def _get_generator(self) -> torch.Generator:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        generator = torch.Generator()
        base_seed = self.seed + worker_id
        if self.sampling_config.seed is not None:
            base_seed = int(self.sampling_config.seed) + worker_id
        generator.manual_seed(base_seed)
        return generator

    def _get_cached_masks(self, mask: torch.Tensor) -> Dict[str, torch.Tensor] | None:
        if not self.sampling_config.enabled or self.sampling_config.mode != "vessel_aware":
            return None
        key = f"{self.sample_id}_{self.sampling_config.vessel_buffer}_{self.sampling_config.background_buffer}_{self.patch_h}x{self.patch_w}"
        if key not in self._mask_cache:
            self._mask_cache[key] = self.sampler.compute_center_masks(mask)
        return self._mask_cache[key]

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        generator = self._get_generator()
        _, height, width = self.image.shape

        if self.mask is None:
            mask = (self.image.mean(dim=0, keepdim=True) > 0).float()
        else:
            mask = self.mask

        mask_2d = mask.squeeze(0)
        cached_masks = self._get_cached_masks(mask_2d)

        while True:
            if self.sampling_config.enabled and self.sampling_config.mode == "vessel_aware":
                cy, cx = self.sampler.sample_center(mask_2d, generator, cached_masks)
                top = cy - self.patch_h // 2
                left = cx - self.patch_w // 2
            else:
                top = torch.randint(0, height - self.patch_h + 1, (1,), generator=generator).item()
                left = torch.randint(0, width - self.patch_w + 1, (1,), generator=generator).item()

            image_patch = self.image[:, top : top + self.patch_h, left : left + self.patch_w]
            mask_patch = mask[:, top : top + self.patch_h, left : left + self.patch_w]

            sample = {"image": image_patch, "mask": mask_patch}
            if self.return_fractal_prior:
                if self.fractal_provider is None:
                    raise ValueError("Fractal prior requested but not configured.")
                patch_box = (top, left, self.patch_h, self.patch_w)
                prior = self.fractal_provider.get_patch(
                    self.sample_id,
                    image_patch=image_patch,
                    image_full=self.image if self.fractal_config.precompute_enabled else None,
                    patch_box=patch_box,
                )
                if prior.ndim == 2:
                    sample["prior"] = prior.unsqueeze(0)
                elif prior.ndim == 3:
                    sample["prior"] = prior
                else:
                    raise ValueError("Prior must be 2D or 3D tensor.")

                if bool(self.fractal_hard_cfg.get("enabled", False)):
                    min_mean = float(self.fractal_hard_cfg.get("min_prior_mean", 0.0))
                    reject_prob = float(self.fractal_hard_cfg.get("reject_prob", 1.0))
                    prior_mean = float(sample["prior"].mean().item())
                    if prior_mean < min_mean:
                        draw = float(torch.rand((), generator=generator).item())
                        if draw < reject_prob:
                            continue

            sample = random_elastic(sample, generator, self.spatial_aug_config.get("elastic", {}))
            sample = random_scale(sample, generator, self.spatial_aug_config.get("scale", {}))
            sample = random_photometric(sample, generator, self.photometric_config)
            sample = random_gaussian_noise(sample, generator, self.spatial_aug_config.get("gaussian_noise", {}))
            sample = random_flip_rotate(sample, generator)
            yield sample


__all__ = ["InfiniteRandomPatchDataset", "PatchCenterSampler", "PatchSamplingConfig", "EpochPatchDataset"]

class EpochPatchDataset(torch.utils.data.Dataset):
    """Map-style dataset that returns exactly one random patch per index.
    Crucially, it caches the base full images and their heavy distance-transform masks
    in lightweight RAM to prevent devastating CPU/IO loops during epoch generation.
    """

    def __init__(
        self,
        base_dataset: torch.utils.data.Dataset,
        indices: list[int],
        length: int,
        patch_size: int | Tuple[int, int] = 128,
        seed: int = 0,
        sampling_config: Dict[str, Any] | None = None,
        fractal_prior_config: Dict[str, Any] | None = None,
        return_fractal_prior: bool = False,
        photometric_config: Dict[str, Any] | None = None,
        max_cached_images: int = 6,
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = indices
        self.length = length
        self.patch_h, self.patch_w = _to_patch_size(patch_size)
        self.seed = seed
        self.sampling_config = _sampling_config_from_dict(sampling_config)
        self.sampler = PatchCenterSampler((self.patch_h, self.patch_w), self.sampling_config)
        self.return_fractal_prior = return_fractal_prior
        self.fractal_config = _fractal_config_from_dict(fractal_prior_config)
        self.fractal_provider = FractalPriorProvider(self.fractal_config) if self.fractal_config.enabled else None
        self.photometric_config = photometric_config or {}
        raw_sampling = sampling_config or {}
        self.fractal_hard_cfg = raw_sampling.get("fractal_hard", {})
        self.spatial_aug_config = (photometric_config or {}).get("spatial", {})

        # --- The Global Memory Cache (LRU-bounded) ---
        self._image_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self._cache_order: list[str] = []  # FIFO eviction order
        self._max_cached_images = max_cached_images
        self._mask_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self.current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch

    def __len__(self) -> int:
        return self.length

    def _get_base_sample(self, idx: int) -> dict[str, Any]:
        """Fetch sample from base dataset, caching with FIFO eviction."""
        base_idx = self.indices[idx % len(self.indices)]
        cache_key = str(base_idx)
        if cache_key not in self._image_cache:
            # Evict oldest if cache is full
            while len(self._image_cache) >= self._max_cached_images and self._cache_order:
                evict_key = self._cache_order.pop(0)
                self._image_cache.pop(evict_key, None)
                # Also evict associated mask caches
                mask_keys = [k for k in self._mask_cache if k.startswith(f"{evict_key}_")]
                for mk in mask_keys:
                    del self._mask_cache[mk]
            sample = self.base_dataset[base_idx]
            self._image_cache[cache_key] = {
                "image": sample["image"],
                "mask": sample["mask"],
                "id": sample.get("id", str(base_idx)),
            }
            self._cache_order.append(cache_key)
        return self._image_cache[cache_key]

    def _get_cached_masks(self, sample_id: str, mask: torch.Tensor) -> Dict[str, torch.Tensor] | None:
        if not self.sampling_config.enabled or self.sampling_config.mode != "vessel_aware":
            return None
        key = f"{sample_id}_{self.sampling_config.vessel_buffer}_{self.sampling_config.background_buffer}_{self.patch_h}x{self.patch_w}"
        if key not in self._mask_cache:
            self._mask_cache[key] = self.sampler.compute_center_masks(mask)
        return self._mask_cache[key]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator()
        global_step = self.current_epoch * self.length + idx
        generator.manual_seed(self.seed + global_step)

        base_sample = self._get_base_sample(idx)
        image = base_sample["image"]
        mask = base_sample["mask"]
        sample_id = base_sample["id"]

        _, height, width = image.shape
        if self.patch_h > height or self.patch_w > width:
            raise ValueError("Patch size must fit within the image.")

        mask_2d = mask.squeeze(0)
        cached_masks = self._get_cached_masks(sample_id, mask_2d)

        while True:
            if self.sampling_config.enabled and self.sampling_config.mode == "vessel_aware":
                cy, cx = self.sampler.sample_center(mask_2d, generator, cached_masks)
                top = cy - self.patch_h // 2
                left = cx - self.patch_w // 2
            else:
                top = torch.randint(0, height - self.patch_h + 1, (1,), generator=generator).item()
                left = torch.randint(0, width - self.patch_w + 1, (1,), generator=generator).item()

            image_patch = image[:, top : top + self.patch_h, left : left + self.patch_w]
            mask_patch = mask[:, top : top + self.patch_h, left : left + self.patch_w]

            sample = {"image": image_patch, "mask": mask_patch}
            if self.return_fractal_prior:
                if self.fractal_provider is None:
                    raise ValueError("Fractal prior requested but not configured.")
                patch_box = (top, left, self.patch_h, self.patch_w)
                prior = self.fractal_provider.get_patch(
                    sample_id,
                    image_patch=image_patch,
                    image_full=image if self.fractal_config.precompute_enabled else None,
                    patch_box=patch_box,
                )
                if prior.ndim == 2:
                    sample["prior"] = prior.unsqueeze(0)
                elif prior.ndim == 3:
                    sample["prior"] = prior

                if bool(self.fractal_hard_cfg.get("enabled", False)):
                    min_mean = float(self.fractal_hard_cfg.get("min_prior_mean", 0.0))
                    reject_prob = float(self.fractal_hard_cfg.get("reject_prob", 1.0))
                    prior_mean = float(sample["prior"].mean().item())
                    if prior_mean < min_mean:
                        if float(torch.rand((), generator=generator).item()) < reject_prob:
                            continue

            sample = random_elastic(sample, generator, self.spatial_aug_config.get("elastic", {}))
            sample = random_scale(sample, generator, self.spatial_aug_config.get("scale", {}))
            sample = random_photometric(sample, generator, self.photometric_config)
            sample = random_gaussian_noise(sample, generator, self.spatial_aug_config.get("gaussian_noise", {}))
            sample = random_flip_rotate(sample, generator)
            return sample
