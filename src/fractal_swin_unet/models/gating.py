"""Fractal gating utilities for skip connections.

Provides two gating mechanisms:
1. apply_fractal_gate — Original hand-crafted sigmoid gate (no learnable params)
2. FractalSPADE — Spatially-Adaptive Prior Modulation (learnable, SPADE-style)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def apply_fractal_gate(
    skip: torch.Tensor,
    lfd_stage: torch.Tensor,
    alpha: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply fractal gating to a skip tensor (original, non-learnable).

    Args:
        skip: Skip tensor of shape (B, C, H, W).
        lfd_stage: LFD map of shape (B, 1, H, W).
        alpha: Gating scale factor.
        eps: Numerical stability term for std.

    Returns:
        Gated skip tensor of shape (B, C, H, W).
    """

    if skip.ndim != 4 or lfd_stage.ndim != 4:
        raise ValueError("skip and lfd_stage must be 4D tensors.")
    if lfd_stage.shape[1] != 1:
        raise ValueError("lfd_stage must have channel dimension 1.")

    mean = lfd_stage.mean(dim=(2, 3), keepdim=True)
    std = lfd_stage.std(dim=(2, 3), keepdim=True)
    normalized = (lfd_stage - mean) / (std + eps)
    gate = torch.sigmoid(alpha * normalized)
    return skip * gate


class FractalSPADE(nn.Module):
    """Spatially-Adaptive Prior Modulation (SPADE-style).

    Instead of a hand-crafted sigmoid gate, this module learns to modulate
    skip-connection features using the fractal prior map. The prior drives
    per-channel, spatially-varying scale (gamma) and shift (beta) through
    instance normalization:

        out = InstanceNorm(skip) * (1 + gamma(prior)) + beta(prior)

    This is strictly more expressive than a multiplicative gate because:
    - It can both amplify AND suppress features (scale)
    - It can add bias to shift feature distributions (shift)
    - It operates per-channel, not just spatially
    - All parameters are learnable

    Reference: Park et al., "Semantic Image Synthesis with Spatially-Adaptive
    Normalization", CVPR 2019.

    Args:
        skip_channels: Number of channels in the skip connection features.
        prior_channels: Number of channels in the fractal prior map (default: 1).
        hidden_dim: Hidden dimension for the shared conv layer.
    """

    def __init__(
        self,
        skip_channels: int,
        prior_channels: int = 1,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm2d(skip_channels, affine=False)
        self.shared = nn.Sequential(
            nn.Conv2d(prior_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.gamma_conv = nn.Conv2d(hidden_dim, skip_channels, kernel_size=3, padding=1)
        self.beta_conv = nn.Conv2d(hidden_dim, skip_channels, kernel_size=3, padding=1)

        # Initialize gamma near 0 so initial behavior ≈ plain InstanceNorm
        nn.init.zeros_(self.gamma_conv.weight)
        nn.init.zeros_(self.gamma_conv.bias)
        nn.init.zeros_(self.beta_conv.weight)
        nn.init.zeros_(self.beta_conv.bias)

    def forward(self, skip: torch.Tensor, lfd: torch.Tensor) -> torch.Tensor:
        """Apply SPADE modulation.

        Args:
            skip: Skip features of shape (B, C, H, W).
            lfd: Fractal prior map of shape (B, prior_channels, H', W').
                 Will be resized to match skip spatial dims if needed.

        Returns:
            Modulated features of shape (B, C, H, W).
        """
        if lfd.shape[2:] != skip.shape[2:]:
            lfd = F.interpolate(lfd, size=skip.shape[2:], mode="bilinear", align_corners=False)

        normalized = self.norm(skip)
        h = self.shared(lfd)
        gamma = self.gamma_conv(h)
        beta = self.beta_conv(h)
        return normalized * (1.0 + gamma) + beta


class FractalSPADEv2(nn.Module):
    """Multi-channel SPADE with deeper shared network.

    Extends FractalSPADE to accept the full 3-channel prior stack from
    the Fractal Anisotropic Diffusion module:
        Channel 0: LFD map (fractal complexity)
        Channel 1: PDE-enhanced image (what to preserve)
        Channel 2: Edge residual (where the boundaries are)

    Key differences from FractalSPADE v1:
    - Accepts multi-channel prior input (default 3)
    - Deeper shared network (2 conv layers with residual)
    - Channel attention on the prior before modulation
    - Residual connection preserving skip identity at init

    Args:
        skip_channels: Number of channels in skip connection features.
        prior_channels: Number of prior channels (default: 3).
        hidden_dim: Hidden dimension for the shared conv layers.
    """

    def __init__(
        self,
        skip_channels: int,
        prior_channels: int = 3,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm2d(skip_channels, affine=False)

        # Deeper shared network: 2 conv layers with residual
        self.shared = nn.Sequential(
            nn.Conv2d(prior_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Channel attention on the shared features
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.Sigmoid(),
        )

        self.gamma_conv = nn.Conv2d(hidden_dim, skip_channels, kernel_size=3, padding=1)
        self.beta_conv = nn.Conv2d(hidden_dim, skip_channels, kernel_size=3, padding=1)

        # Initialize near identity (gamma≈0, beta≈0 → output ≈ InstanceNorm(skip))
        nn.init.zeros_(self.gamma_conv.weight)
        nn.init.zeros_(self.gamma_conv.bias)  # type: ignore[arg-type]
        nn.init.zeros_(self.beta_conv.weight)
        nn.init.zeros_(self.beta_conv.bias)  # type: ignore[arg-type]

    def forward(self, skip: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        """Apply multi-channel SPADE modulation.

        Args:
            skip: Skip features (B, C, H, W).
            prior: Multi-channel prior (B, prior_channels, H', W').
                   Resized to match skip spatial dims if needed.

        Returns:
            Modulated features (B, C, H, W).
        """
        if prior.shape[2:] != skip.shape[2:]:
            prior = F.interpolate(
                prior, size=skip.shape[2:], mode="bilinear", align_corners=False
            )

        normalized = self.norm(skip)

        # Shared feature extraction
        h = self.shared(prior)  # (B, hidden, H, W)

        # Channel attention
        attn = self.channel_attn(h)  # (B, hidden)
        h = h * attn.unsqueeze(-1).unsqueeze(-1)  # (B, hidden, H, W)

        # Modulation parameters
        gamma = self.gamma_conv(h)
        beta = self.beta_conv(h)

        return normalized * (1.0 + gamma) + beta
