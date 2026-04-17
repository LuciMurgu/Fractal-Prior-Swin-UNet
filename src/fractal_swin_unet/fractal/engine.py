"""Fractal prior engine interfaces and implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import torch

from .lfd import compute_lfd_map, LFDParams


class FractalPriorEngine:
    """Abstract fractal prior engine interface."""

    name: str = "base"

    def params_signature(self) -> Dict[str, object]:
        raise NotImplementedError

    def compute_full(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def compute_patch(self, image_patch: torch.Tensor) -> torch.Tensor:
        return self.compute_full(image_patch)


@dataclass
class DbcLfdConfig:
    window: int = 15
    stride: int = 4
    box_sizes: Iterable[int] = (2, 4, 8, 16)
    fast_mode: bool = True


class DbcLfdEngine(FractalPriorEngine):
    name = "dbc_lfd"

    def __init__(self, config: DbcLfdConfig) -> None:
        self.config = config

    def params_signature(self) -> Dict[str, object]:
        return {
            "engine": self.name,
            "lfd.window": self.config.window,
            "lfd.stride": self.config.stride,
            "lfd.box_sizes": list(self.config.box_sizes),
            "lfd.fast_mode": self.config.fast_mode,
        }

    def compute_full(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 3:
            image = image.mean(dim=0)
        params = LFDParams(
            window_size=self.config.window,
            stride=self.config.stride,
            box_sizes=tuple(self.config.box_sizes),
            fast_mode=self.config.fast_mode,
        )
        return compute_lfd_map(image, params=params)
