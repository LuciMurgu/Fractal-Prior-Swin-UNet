"""Auxiliary decoders for edge and skeleton prediction (ECCV 2024 MD-Net).

These lightweight decoder heads share the encoder and force it to learn
topology-aware features by predicting vessel edges and skeletons as
auxiliary tasks during training.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .swin_unet import SwinBlock


class EdgeDecoder(nn.Module):
    """Lightweight decoder that predicts vessel boundaries.

    Edge GT is extracted from the vessel mask via morphological gradient
    (dilation - erosion).
    """

    def __init__(self, in_channels: int, drop_rate: float = 0.0) -> None:
        super().__init__()
        mid = max(in_channels // 2, 8)
        self.conv1 = nn.Conv2d(in_channels, mid, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, 1, kernel_size=1)
        self.drop = nn.Dropout2d(drop_rate)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.bn1(self.conv1(features)))
        x = self.drop(x)
        return self.conv2(x)


class SkeletonDecoder(nn.Module):
    """Lightweight decoder that predicts vessel skeletons/centerlines.

    Skeleton GT is extracted from the vessel mask via morphological
    skeletonization.
    """

    def __init__(self, in_channels: int, drop_rate: float = 0.0) -> None:
        super().__init__()
        mid = max(in_channels // 2, 8)
        self.conv1 = nn.Conv2d(in_channels, mid, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, 1, kernel_size=1)
        self.drop = nn.Dropout2d(drop_rate)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.bn1(self.conv1(features)))
        x = self.drop(x)
        return self.conv2(x)


class FDRegressionHead(nn.Module):
    """Auxiliary head that predicts fractal dimension from bottleneck features.

    Forces the encoder to encode structural/morphological information
    in the latent representation (Fractal-Driven Regularization).

    The target is the precomputed FD of the ground-truth vessel mask patch.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.GELU(),
            nn.Linear(in_channels // 2, 1),
        )

    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """Predict scalar FD from bottleneck features.

        Args:
            bottleneck: (B, C, H, W) bottleneck feature map.

        Returns:
            (B, 1) predicted fractal dimension.
        """
        x = self.pool(bottleneck).flatten(1)
        return self.fc(x)


# --- GT extraction utilities ---

def extract_edge_gt(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Extract edge GT from binary mask via morphological gradient.

    Args:
        mask: (B, 1, H, W) binary vessel mask.
        kernel_size: Size of the morphological kernel.

    Returns:
        (B, 1, H, W) binary edge map.
    """
    mask = mask.float()
    pad = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    eroded = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=pad)
    edge = (dilated - eroded).clamp(0, 1)
    return edge


def extract_skeleton_gt(mask: torch.Tensor, iters: int = 8) -> torch.Tensor:
    """Extract soft skeleton GT via iterative morphological thinning.

    Uses the same soft_skeletonize approach as the loss function
    for consistency — avoids needing skimage at training time.

    Args:
        mask: (B, 1, H, W) binary vessel mask.
        iters: Number of thinning iterations.

    Returns:
        (B, 1, H, W) soft skeleton map.
    """
    mask = mask.float().clamp(0, 1)

    def _soft_erode(x: torch.Tensor) -> torch.Tensor:
        return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)

    def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)

    def _soft_open(x: torch.Tensor) -> torch.Tensor:
        return _soft_dilate(_soft_erode(x))

    skel = F.relu(mask - _soft_open(mask))
    x = mask.clone()
    for _ in range(max(iters - 1, 0)):
        x = _soft_erode(x)
        delta = F.relu(x - _soft_open(x))
        skel = skel + F.relu(delta - skel * delta)

    return skel.clamp(0, 1)
