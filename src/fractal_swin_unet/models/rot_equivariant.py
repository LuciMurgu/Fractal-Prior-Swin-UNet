"""Rotation-Equivariant Convolution via Steerable Filter Bank.

Replaces standard depthwise convolution with a rotation-equivariant
version that detects features at ALL orientations using a SINGLE learned
filter, then max-pools over orientations for invariant response.

For retinal vessels: a filter that detects "horizontal vessel edge" will
automatically detect it at any angle (15°, 45°, 90°, ...) without
needing separate weights for each orientation. This is the inductive
bias that standard convolutions fundamentally cannot learn.

Key insight from RSF-Conv (Sun et al., 2024): rotation equivariance
via Fourier parameterization. We implement a practical variant:
- Learn ONE base filter per channel
- Generate N rotated copies via affine_grid + grid_sample (differentiable)
- Max-pool over rotation dimension → orientation-invariant response
- Optional: keep per-orientation responses for downstream processing

Complexity: O(N × standard conv) where N = number of orientations.
With N=8, this is 8× more FLOPs per conv but with the SAME parameter count.

Reference:
    Sun et al., "RSF-Conv: Rotation-Scale Fourier Convolution", arXiv 2024.
    Cohen & Welling, "Group Equivariant CNNs", ICML 2016.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


def _rotate_kernel(
    weight: torch.Tensor,
    angle_deg: float,
) -> torch.Tensor:
    """Rotate a conv2d kernel by angle_deg degrees (deterministic).

    Uses manual bilinear interpolation on the small kernel grid instead
    of F.grid_sample, avoiding the non-deterministic backward pass issue.

    Args:
        weight: (out_channels, in_channels, H, W) conv kernel.
        angle_deg: Rotation angle in degrees.

    Returns:
        Rotated kernel of same shape.
    """
    if abs(angle_deg) < 1e-6:
        return weight

    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    oc, ic, kh, kw = weight.shape

    # Create coordinate grid for the kernel (centered at origin)
    cy, cx = (kh - 1) / 2.0, (kw - 1) / 2.0
    ys = torch.arange(kh, device=weight.device, dtype=weight.dtype) - cy
    xs = torch.arange(kw, device=weight.device, dtype=weight.dtype) - cx
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    # Inverse rotation: for each output position, find source position
    src_x = cos_a * grid_x + sin_a * grid_y + cx
    src_y = -sin_a * grid_x + cos_a * grid_y + cy

    # Bilinear interpolation (manual, fully deterministic)
    x0 = src_x.long()
    y0 = src_y.long()
    x1 = x0 + 1
    y1 = y0 + 1

    # Fractional parts
    fx = (src_x - x0.float()).clamp(0, 1)
    fy = (src_y - y0.float()).clamp(0, 1)

    # Boundary mask (zero-pad out-of-bounds)
    def _safe(val: torch.Tensor, limit: int) -> tuple[torch.Tensor, torch.Tensor]:
        valid = (val >= 0) & (val < limit)
        clamped = val.clamp(0, limit - 1)
        return clamped, valid.float()

    x0c, vx0 = _safe(x0, kw)
    y0c, vy0 = _safe(y0, kh)
    x1c, vx1 = _safe(x1, kw)
    y1c, vy1 = _safe(y1, kh)

    # Reshape weight for indexing: (oc*ic, kh, kw)
    w = weight.reshape(-1, kh, kw)

    # Gather the 4 neighbors (using advanced indexing)
    batch_idx = torch.arange(w.shape[0], device=weight.device).view(-1, 1, 1)
    batch_idx = batch_idx.expand(-1, kh, kw)

    q00 = w[batch_idx, y0c, x0c] * (vx0 * vy0)
    q01 = w[batch_idx, y0c, x1c] * (vx1 * vy0)
    q10 = w[batch_idx, y1c, x0c] * (vx0 * vy1)
    q11 = w[batch_idx, y1c, x1c] * (vx1 * vy1)

    # Bilinear interpolation
    result = (
        q00 * (1 - fx) * (1 - fy)
        + q01 * fx * (1 - fy)
        + q10 * (1 - fx) * fy
        + q11 * fx * fy
    )

    return result.reshape(oc, ic, kh, kw)


class RotEquivariantConv2d(nn.Module):
    """Rotation-equivariant convolution via rotated filter bank.

    Learns ONE base filter, generates N rotated versions, applies all,
    and aggregates (max or mean) over orientations.

    For depthwise convolution (groups=channels), each channel gets its
    own rotation-equivariant filter bank.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Filter size (only odd sizes supported).
        n_rotations: Number of discrete rotation samples.
        padding: Padding size.
        groups: Number of groups (set = in_channels for depthwise).
        aggregation: How to combine rotated responses ('max' or 'mean').
        bias: Whether to include a bias term.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        n_rotations: int = 8,
        padding: int = 1,
        groups: int = 1,
        aggregation: Literal["max", "mean"] = "max",
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_rotations = n_rotations
        self.padding = padding
        self.groups = groups
        self.aggregation = aggregation

        # Learn ONE base filter
        self.base_weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

        # Precompute rotation angles (evenly spaced)
        self.angles = [i * 360.0 / n_rotations for i in range(n_rotations)]

        # Cache for rotated kernels (cleared on weight update)
        self._cached_kernels: list[torch.Tensor] | None = None

    def _get_rotated_kernels(self) -> list[torch.Tensor]:
        """Generate all rotated versions of the base kernel."""
        # Don't cache during training (weights change)
        if self.training or self._cached_kernels is None:
            kernels = [_rotate_kernel(self.base_weight, angle) for angle in self.angles]
            if not self.training:
                self._cached_kernels = kernels
            return kernels
        return self._cached_kernels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotation-equivariant convolution.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Output tensor (B, out_channels, H', W') after max/mean pooling
            over all orientations.
        """
        kernels = self._get_rotated_kernels()

        # Apply each rotated kernel
        responses = []
        for kernel in kernels:
            r = F.conv2d(x, kernel, bias=None, padding=self.padding, groups=self.groups)
            responses.append(r)

        # Stack: (N, B, C, H, W)
        stacked = torch.stack(responses, dim=0)

        # Aggregate over rotation dimension
        if self.aggregation == "max":
            out = stacked.max(dim=0).values
        else:
            out = stacked.mean(dim=0)

        # Add bias
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)

        return out

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, n_rotations={self.n_rotations}, "
            f"groups={self.groups}, aggregation={self.aggregation}"
        )


class RotEquivariantSwinBlock(nn.Module):
    """Swin-style block with rotation-equivariant depthwise convolution.

    Drop-in replacement for SwinBlock. The depthwise conv is replaced with
    RotEquivariantConv2d, making the spatial mixing rotation-invariant.

    This means vessel features are detected identically at ALL orientations
    using the SAME learned filter — a fundamental inductive bias improvement
    for tubular structures like retinal vessels.

    Args:
        channels: Feature channels.
        n_rotations: Number of discrete rotations (default 8 = every 45°).
        drop_rate: Dropout rate.
        aggregation: 'max' for orientation-invariant, 'mean' for average.
    """

    def __init__(
        self,
        channels: int,
        n_rotations: int = 8,
        drop_rate: float = 0.0,
        aggregation: Literal["max", "mean"] = "max",
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)

        # Rotation-equivariant depthwise conv
        self.dwconv = RotEquivariantConv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            n_rotations=n_rotations,
            aggregation=aggregation,
        )

        self.drop1 = nn.Dropout2d(drop_rate) if drop_rate > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(channels)

        mlp_layers: list[nn.Module] = [
            nn.Conv2d(channels, channels * 4, kernel_size=1),
            nn.GELU(),
        ]
        if drop_rate > 0:
            mlp_layers.append(nn.Dropout(drop_rate))
        mlp_layers.append(nn.Conv2d(channels * 4, channels, kernel_size=1))
        if drop_rate > 0:
            mlp_layers.append(nn.Dropout(drop_rate))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from .swin_unet import _layer_norm_2d

        x = x + self.drop1(self.dwconv(_layer_norm_2d(x, self.norm1)))
        x = x + self.mlp(_layer_norm_2d(x, self.norm2))
        return x


class RotEquivariantSwinStage(nn.Module):
    """Stacked rotation-equivariant Swin blocks with optional downsample."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        downsample: bool,
        n_rotations: int = 8,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.downsample = None
        if downsample or in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.blocks = nn.Sequential(
            *[
                RotEquivariantSwinBlock(
                    out_channels,
                    n_rotations=n_rotations,
                    drop_rate=drop_rate,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample is not None:
            x = self.downsample(x)
        return self.blocks(x)
