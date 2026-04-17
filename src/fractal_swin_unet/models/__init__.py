"""Model registry for baseline architectures."""

from .fractal_nca import FractalNCA
from .fractal_prior_swin_unet import FractalPriorSwinUNet
from .swin_unet import SwinUNetTiny

__all__ = ["FractalNCA", "FractalPriorSwinUNet", "SwinUNetTiny"]
