"""Shim package to allow local execution without installation."""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
_src_pkg = Path(__file__).resolve().parent.parent / "src" / "fractal_swin_unet"
if _src_pkg.exists():
    __path__.append(str(_src_pkg))

from .seed import set_deterministic_seed  # noqa: E402
from .fractal import LFDCache, LFDParams, compute_lfd_map  # noqa: E402
from .model import SimpleSwinUNet  # noqa: E402
from .models import FractalPriorSwinUNet, SwinUNetTiny  # noqa: E402
from .data import InfiniteRandomPatchDataset  # noqa: E402
from .losses import dice_bce_loss  # noqa: E402

__all__ = [
    "set_deterministic_seed",
    "compute_lfd_map",
    "LFDCache",
    "LFDParams",
    "SimpleSwinUNet",
    "FractalPriorSwinUNet",
    "SwinUNetTiny",
    "InfiniteRandomPatchDataset",
    "dice_bce_loss",
]
