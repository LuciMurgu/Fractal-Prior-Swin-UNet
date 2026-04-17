"""Frequency-Geometric Decomposition via Haar DWT (inspired by FGOS-Net, 2026).

Decomposes feature maps into low-frequency (stable topology carrier) and
high-frequency (directional boundary details) components using the Haar
Discrete Wavelet Transform. The recombination uses learnable gating to
let the network decide how much boundary detail to inject.

This is a zero-cost geometric prior — the Haar transform requires no
learned parameters and adds negligible compute.

Reference: FGOS-Net (2026) — "Frequency-Geometric Disentanglement for
thin structure segmentation via geometry-conditioned state space models."
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def haar_dwt_2d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """2D Haar Discrete Wavelet Transform.

    Args:
        x: Input tensor of shape (B, C, H, W). H and W should be even.

    Returns:
        (LL, LH, HL, HH) — each of shape (B, C, H//2, W//2).
        LL: Low-freq approximation (stable topology).
        LH: Horizontal detail (vertical edges).
        HL: Vertical detail (horizontal edges).
        HH: Diagonal detail.
    """
    # Pad if odd dimensions
    _, _, h, w = x.shape
    if h % 2 != 0:
        x = F.pad(x, (0, 0, 0, 1), mode="reflect")
    if w % 2 != 0:
        x = F.pad(x, (0, 1, 0, 0), mode="reflect")

    # Separate even/odd indices
    x_even_h = x[:, :, 0::2, :]  # even rows
    x_odd_h = x[:, :, 1::2, :]   # odd rows

    # Low and high frequency along rows
    L = (x_even_h + x_odd_h) * 0.5
    H = (x_even_h - x_odd_h) * 0.5

    # Now along columns
    LL = (L[:, :, :, 0::2] + L[:, :, :, 1::2]) * 0.5
    LH = (L[:, :, :, 0::2] - L[:, :, :, 1::2]) * 0.5
    HL = (H[:, :, :, 0::2] + H[:, :, :, 1::2]) * 0.5
    HH = (H[:, :, :, 0::2] - H[:, :, :, 1::2]) * 0.5

    return LL, LH, HL, HH


def haar_idwt_2d(
    LL: torch.Tensor,
    LH: torch.Tensor,
    HL: torch.Tensor,
    HH: torch.Tensor,
) -> torch.Tensor:
    """2D Inverse Haar Discrete Wavelet Transform.

    Args:
        LL, LH, HL, HH: Wavelet components, each (B, C, H//2, W//2).

    Returns:
        Reconstructed tensor of shape (B, C, H, W).
    """
    # Reconstruct columns
    L_even = LL + LH
    L_odd = LL - LH
    H_even = HL + HH
    H_odd = HL - HH

    # Interleave columns
    b, c, h2, w2 = LL.shape
    L = torch.zeros(b, c, h2, w2 * 2, device=LL.device, dtype=LL.dtype)
    L[:, :, :, 0::2] = L_even
    L[:, :, :, 1::2] = L_odd

    H = torch.zeros(b, c, h2, w2 * 2, device=LL.device, dtype=LL.dtype)
    H[:, :, :, 0::2] = H_even
    H[:, :, :, 1::2] = H_odd

    # Reconstruct rows
    x_even = L + H
    x_odd = L - H

    x = torch.zeros(b, c, h2 * 2, w2 * 2, device=LL.device, dtype=LL.dtype)
    x[:, :, 0::2, :] = x_even
    x[:, :, 1::2, :] = x_odd

    return x


class FreqGeomDecomposition(nn.Module):
    """Frequency-Geometric Decomposition module.

    Decomposes input features into topology (low-freq) and boundary
    (high-freq) components, processes them with learnable gating,
    and recombines. Zero-overhead from Haar transform; only learnable
    params are the gating weights.

    Can be inserted at any decoder stage to improve thin-vessel
    sensitivity without adding significant complexity.
    """

    def __init__(self, channels: int, high_freq_boost: float = 1.0) -> None:
        """
        Args:
            channels: Number of input/output feature channels.
            high_freq_boost: Initial scaling for high-frequency injection.
                Higher values emphasize boundary details.
        """
        super().__init__()
        self.channels = channels

        # Learnable mixing weights for topology vs boundary
        self.topo_gate = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.boundary_gate = nn.Parameter(
            torch.ones(1, channels, 1, 1) * high_freq_boost
        )

        # 1x1 conv to fuse high-freq components (LH + HL + HH → single map)
        self.hf_fuse = nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False)

        # Directional attention: learn which high-freq band matters most
        self.dir_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels * 3, 3),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) feature map.

        Returns:
            (B, C, H, W) — recombined features with enhanced boundaries.
        """
        identity = x

        # Decompose
        LL, LH, HL, HH = haar_dwt_2d(x)

        # Directional attention over high-freq bands
        hf_concat = torch.cat([LH, HL, HH], dim=1)  # (B, 3C, H/2, W/2)
        dir_weights = self.dir_attn(hf_concat)  # (B, 3)

        # Weight each directional band
        b = dir_weights.shape[0]
        LH_w = LH * dir_weights[:, 0].view(b, 1, 1, 1)
        HL_w = HL * dir_weights[:, 1].view(b, 1, 1, 1)
        HH_w = HH * dir_weights[:, 2].view(b, 1, 1, 1)

        # Fuse high-freq
        hf_weighted = torch.cat([LH_w, HL_w, HH_w], dim=1)
        hf_fused = self.hf_fuse(hf_weighted)  # (B, C, H/2, W/2)

        # Reconstruct with learnable gating
        LL_gated = LL * self.topo_gate
        recon = haar_idwt_2d(LL_gated, hf_fused, torch.zeros_like(HL), torch.zeros_like(HH))

        # Match original size (in case of odd padding)
        if recon.shape != identity.shape:
            recon = recon[:, :, :identity.shape[2], :identity.shape[3]]

        # Residual connection
        return identity + self.boundary_gate * recon
