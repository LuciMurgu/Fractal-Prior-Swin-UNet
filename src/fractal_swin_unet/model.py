"""Minimal UNet-style model used for smoke tests."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Two-layer convolutional block with ReLU activations."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleSwinUNet(nn.Module):
    """UNet-style model with optional LFD gating at skip connections."""

    def __init__(self, in_channels: int = 1, base_channels: int = 16) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = ConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = ConvBlock(base_channels * 2 + base_channels, base_channels)

        self.out_conv = nn.Conv2d(base_channels, 1, kernel_size=1)

    def _gate_skip(self, skip: torch.Tensor, lfd_map: Optional[torch.Tensor]) -> torch.Tensor:
        if lfd_map is None:
            return skip
        if lfd_map.shape[2:] != skip.shape[2:]:
            lfd_map = nn.functional.interpolate(lfd_map, size=skip.shape[2:], mode="bilinear", align_corners=False)
        gate = torch.sigmoid(lfd_map)
        return skip * gate

    def forward(self, x: torch.Tensor, lfd_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        enc2 = self.enc2(x)
        x = self.pool2(enc2)

        x = self.bottleneck(x)

        x = self.up2(x)
        enc2_gated = self._gate_skip(enc2, lfd_map)
        x = torch.cat([x, enc2_gated], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        enc1_gated = self._gate_skip(enc1, lfd_map)
        x = torch.cat([x, enc1_gated], dim=1)
        x = self.dec1(x)

        return self.out_conv(x)
