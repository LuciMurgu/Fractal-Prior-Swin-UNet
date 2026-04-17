"""Lightweight Swin-UNet baseline for binary segmentation."""

from __future__ import annotations

import torch
from torch import nn


def _layer_norm_2d(x: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
    b, c, h, w = x.shape
    x = x.permute(0, 2, 3, 1)
    x = norm(x)
    return x.permute(0, 3, 1, 2).reshape(b, c, h, w)


class SwinBlock(nn.Module):
    """Minimal Swin-style block using depthwise conv + MLP."""

    def __init__(self, channels: int, drop_rate: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
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
        x = x + self.drop1(self.dwconv(_layer_norm_2d(x, self.norm1)))
        x = x + self.mlp(_layer_norm_2d(x, self.norm2))
        return x


class SwinStage(nn.Module):
    """Stacked Swin blocks with optional downsample."""

    def __init__(self, in_channels: int, out_channels: int, depth: int, downsample: bool, drop_rate: float = 0.0) -> None:
        super().__init__()
        self.downsample = None
        if downsample or in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.blocks = nn.Sequential(*[SwinBlock(out_channels, drop_rate=drop_rate) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample is not None:
            x = self.downsample(x)
        return self.blocks(x)


class SwinUNetTiny(nn.Module):
    """Tiny Swin-UNet baseline suitable for CPU smoke tests."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 32,
        depths: tuple[int, int, int] = (1, 1, 1),
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.stem = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)

        self.enc1 = SwinStage(embed_dim, embed_dim, depths[0], downsample=False, drop_rate=drop_rate)
        self.enc2 = SwinStage(embed_dim, embed_dim * 2, depths[1], downsample=True, drop_rate=drop_rate)
        self.enc3 = SwinStage(embed_dim * 2, embed_dim * 4, depths[2], downsample=True, drop_rate=drop_rate)

        self.bottleneck = SwinBlock(embed_dim * 4, drop_rate=drop_rate)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(embed_dim * 4 + embed_dim * 2, embed_dim * 2, kernel_size=3, padding=1),
            SwinBlock(embed_dim * 2, drop_rate=drop_rate),
        )

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(embed_dim * 2 + embed_dim, embed_dim, kernel_size=3, padding=1),
            SwinBlock(embed_dim, drop_rate=drop_rate),
        )

        self.out_conv = nn.Conv2d(embed_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)

        x = self.bottleneck(s3)

        x = self.up2(x)
        x = torch.cat([x, s2], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, s1], dim=1)
        x = self.dec1(x)

        return self.out_conv(x)
