"""Frangi vesselness filter via Hessian eigenvalue analysis.

Implements multi-scale tubular structure detection using the PDE:
    H(σ) = σ² · [∂²G_σ/∂x_i∂x_j] * I

where G_σ is a Gaussian kernel at scale σ. The vesselness is derived
from the eigenvalues of the Hessian matrix at each scale.

Reference: Frangi et al., "Multiscale vessel enhancement filtering" (MICCAI 1998)
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F


def _gaussian_kernel_1d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create a 1D Gaussian kernel."""
    radius = int(math.ceil(3.0 * sigma))
    size = 2 * radius + 1
    x = torch.arange(size, device=device, dtype=dtype) - radius
    kernel = torch.exp(-x ** 2 / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply separable Gaussian blur."""
    if sigma < 0.5:
        return x
    kernel = _gaussian_kernel_1d(sigma, x.device, x.dtype)
    k = kernel.numel()
    pad = k // 2
    # Separable: blur rows then columns
    kx = kernel.view(1, 1, 1, k)
    ky = kernel.view(1, 1, k, 1)
    out = F.pad(x, (pad, pad, 0, 0), mode="reflect")
    out = F.conv2d(out, kx.expand(x.shape[1], -1, -1, -1), groups=x.shape[1])
    out = F.pad(out, (0, 0, pad, pad), mode="reflect")
    out = F.conv2d(out, ky.expand(x.shape[1], -1, -1, -1), groups=x.shape[1])
    return out


def _hessian_2d(x: torch.Tensor, sigma: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute scale-normalized Hessian components.
    
    Returns:
        (Ixx, Ixy, Iyy) each of shape (B, 1, H, W), scaled by σ².
    """
    blurred = _gaussian_blur(x, sigma)
    
    # Second derivatives via finite differences
    padded = F.pad(blurred, (1, 1, 1, 1), mode="reflect")
    
    # ∂²I/∂x²
    Ixx = padded[:, :, 1:-1, 2:] - 2 * blurred + padded[:, :, 1:-1, :-2]
    # ∂²I/∂y²
    Iyy = padded[:, :, 2:, 1:-1] - 2 * blurred + padded[:, :, :-2, 1:-1]
    # ∂²I/∂x∂y
    Ixy = (padded[:, :, 2:, 2:] - padded[:, :, 2:, :-2]
           - padded[:, :, :-2, 2:] + padded[:, :, :-2, :-2]) / 4.0
    
    # Scale normalization
    scale = sigma ** 2
    return Ixx * scale, Ixy * scale, Iyy * scale


def _eigenvalues_2d(
    Ixx: torch.Tensor, Ixy: torch.Tensor, Iyy: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute sorted eigenvalues of symmetric 2x2 Hessian.
    
    Returns (λ1, λ2) where |λ1| ≤ |λ2|.
    """
    trace = Ixx + Iyy
    det = Ixx * Iyy - Ixy ** 2
    discriminant = torch.sqrt(torch.clamp(trace ** 2 - 4 * det, min=0.0))
    
    e1 = (trace + discriminant) / 2.0
    e2 = (trace - discriminant) / 2.0
    
    # Sort by absolute value: |λ1| ≤ |λ2|
    abs1 = e1.abs()
    abs2 = e2.abs()
    lam1 = torch.where(abs1 <= abs2, e1, e2)
    lam2 = torch.where(abs1 <= abs2, e2, e1)
    return lam1, lam2


def frangi_vesselness(
    image: torch.Tensor,
    sigmas: Sequence[float] = (1.0, 2.0, 4.0),
    beta: float = 0.5,
    c: float = 15.0,
    black_ridges: bool = True,
) -> torch.Tensor:
    """Compute Frangi vesselness filter response.

    Args:
        image: Input tensor of shape (C, H, W) or (B, C, H, W).
        sigmas: Scales for multi-scale analysis.
        beta: Blobness suppression parameter.
        c: Structureness sensitivity (auto-scaled if 0).
        black_ridges: If True, detect dark vessels on bright background.

    Returns:
        Vesselness map of shape (1, H, W) or (B, 1, H, W), in [0, 1].
    """
    squeeze = False
    if image.ndim == 3:
        image = image.unsqueeze(0)
        squeeze = True

    # Convert to grayscale
    if image.shape[1] == 3:
        gray = 0.2989 * image[:, 0:1] + 0.5870 * image[:, 1:2] + 0.1140 * image[:, 2:3]
    elif image.shape[1] == 1:
        gray = image
    else:
        gray = image.mean(dim=1, keepdim=True)

    max_vesselness = torch.zeros_like(gray)

    for sigma in sigmas:
        Ixx, Ixy, Iyy = _hessian_2d(gray, sigma)
        lam1, lam2 = _eigenvalues_2d(Ixx, Ixy, Iyy)

        # For vessels (bright on dark or dark on bright):
        # λ1 ≈ 0, |λ2| >> 0, and λ2 < 0 for bright vessels
        if black_ridges:
            # Dark vessels on bright background: λ2 > 0
            valid = lam2 > 0
        else:
            # Bright vessels: λ2 < 0
            valid = lam2 < 0
            lam2 = lam2.abs()
            lam1 = lam1.abs()

        # Blobness ratio R_B = |λ1| / |λ2|
        R_B = lam1.abs() / (lam2.abs() + 1e-8)

        # Structureness S = sqrt(λ1² + λ2²)
        S = torch.sqrt(lam1 ** 2 + lam2 ** 2 + 1e-8)

        # Vesselness
        V = torch.exp(-R_B ** 2 / (2 * beta ** 2)) * (1.0 - torch.exp(-S ** 2 / (2 * c ** 2)))
        V = V * valid.float()

        max_vesselness = torch.maximum(max_vesselness, V)

    # Normalize to [0, 1]
    vmax = max_vesselness.amax(dim=(-2, -1), keepdim=True)
    result = max_vesselness / (vmax + 1e-8)

    if squeeze:
        result = result.squeeze(0)
    return result
