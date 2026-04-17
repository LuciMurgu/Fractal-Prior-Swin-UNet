"""Patch center sampling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def _to_hw(patch_size: int | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(patch_size, int):
        return patch_size, patch_size
    return patch_size


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel = 2 * radius + 1
    pooled = F.max_pool2d(mask.float(), kernel_size=kernel, stride=1, padding=radius)
    return pooled > 0


def _valid_center_mask(height: int, width: int, patch_h: int, patch_w: int) -> torch.Tensor:
    center = torch.zeros((height, width), dtype=torch.bool)
    cy_min = patch_h // 2
    cx_min = patch_w // 2
    cy_max = height - (patch_h - patch_h // 2)
    cx_max = width - (patch_w - patch_w // 2)
    center[cy_min:cy_max, cx_min:cx_max] = True
    return center


@dataclass
class PatchSamplingConfig:
    enabled: bool = False
    mode: str = "uniform"
    p_vessel: float = 0.7
    p_background: float = 0.3
    vessel_buffer: int = 3
    background_buffer: int = 0
    min_vessel_fraction_in_patch: float = 0.0
    max_retries: int = 30
    seed: int | None = None


class PatchCenterSampler:
    """Sample patch centers with optional vessel-aware bias."""

    def __init__(self, patch_size: int | Tuple[int, int], config: PatchSamplingConfig) -> None:
        self.patch_h, self.patch_w = _to_hw(patch_size)
        self.config = config

    def compute_center_masks(self, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        if mask.ndim != 2:
            raise ValueError("mask must be 2D (H, W).")
        height, width = mask.shape
        vessel_pixels = mask > 0
        near_vessel = _dilate(vessel_pixels.unsqueeze(0).unsqueeze(0), self.config.vessel_buffer).squeeze(0).squeeze(0)
        if self.config.background_buffer > 0:
            background = ~_dilate(vessel_pixels.unsqueeze(0).unsqueeze(0), self.config.background_buffer).squeeze(0).squeeze(0)
        else:
            background = ~vessel_pixels

        valid_centers = _valid_center_mask(height, width, self.patch_h, self.patch_w)
        vessel_centers = near_vessel & valid_centers
        background_centers = background & valid_centers

        return {
            "valid_centers": valid_centers,
            "vessel_centers": vessel_centers,
            "background_centers": background_centers,
        }

    def sample_center(
        self,
        mask: torch.Tensor,
        generator: torch.Generator,
        cached_masks: Dict[str, torch.Tensor] | None = None,
    ) -> Tuple[int, int]:
        if mask.ndim != 2:
            raise ValueError("mask must be 2D (H, W).")

        height, width = mask.shape
        if cached_masks is None:
            cached_masks = self.compute_center_masks(mask)

        vessel_centers = cached_masks["vessel_centers"]
        background_centers = cached_masks["background_centers"]
        valid_centers = cached_masks["valid_centers"]

        p_vessel = max(self.config.p_vessel, 0.0)
        p_background = max(self.config.p_background, 0.0)
        total = p_vessel + p_background
        if total <= 0:
            p_vessel = 0.0
        else:
            p_vessel = p_vessel / total

        for _ in range(self.config.max_retries):
            use_vessel = bool(torch.rand((), generator=generator).item() < p_vessel)
            chosen = vessel_centers if use_vessel else background_centers
            if not chosen.any():
                chosen = background_centers if use_vessel else vessel_centers
            if not chosen.any():
                chosen = valid_centers

            center_idx = _sample_from_mask(chosen, generator)
            if center_idx is None:
                center_idx = _sample_from_mask(valid_centers, generator)
            cy, cx = center_idx

            if self.config.min_vessel_fraction_in_patch > 0:
                top = cy - self.patch_h // 2
                left = cx - self.patch_w // 2
                patch = mask[top : top + self.patch_h, left : left + self.patch_w]
                frac = float(patch.float().mean().item())
                if frac < self.config.min_vessel_fraction_in_patch:
                    continue

            return cy, cx

        center_idx = _sample_from_mask(valid_centers, generator)
        if center_idx is None:
            raise ValueError("No valid centers available for sampling.")
        return center_idx


def _sample_from_mask(mask: torch.Tensor, generator: torch.Generator) -> Tuple[int, int] | None:
    indices = mask.nonzero(as_tuple=False)
    if indices.numel() == 0:
        return None
    idx = torch.randint(0, indices.shape[0], (1,), generator=generator).item()
    cy, cx = indices[idx].tolist()
    return int(cy), int(cx)
