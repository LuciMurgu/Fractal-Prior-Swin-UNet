"""Fractal prior utilities (DBC/LFD/Lacunarity/Percolation/Multi-scale FFM)."""

from .cache import LFDCache
from .dbc import dbc_fractal_dimension
from .lacunarity import compute_lacunarity_map, gliding_box_lacunarity
from .lfd import LFDParams, compute_lfd_map, compute_multiscale_ffm
from .percolation import compute_percolation_map, percolation_critical_threshold

__all__ = [
    "LFDCache",
    "LFDParams",
    "dbc_fractal_dimension",
    "compute_lfd_map",
    "compute_multiscale_ffm",
    "compute_lacunarity_map",
    "gliding_box_lacunarity",
    "compute_percolation_map",
    "percolation_critical_threshold",
]

