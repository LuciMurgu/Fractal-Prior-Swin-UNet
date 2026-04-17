"""Hessian Direction Field — domain-agnostic vessel orientation prior.

Computes the primary eigenvector (vessel tangent) of the Hessian matrix
at each pixel, producing a 2-channel orientation field (cos θ, sin θ)
that captures vessel tangent direction.

This is a domain-agnostic representation — the direction field transfers
across imaging modalities (fundus → OCTA) because vessel orientation is
a geometric property invariant to imaging physics.

Reference: Hu et al., "Hessian-based Vector Field Transformer",
Medical Image Analysis 2024.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F


def _gaussian_kernel_1d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create a 1D Gaussian kernel."""
    radius = max(1, int(math.ceil(3.0 * sigma)))
    size = 2 * radius + 1
    x = torch.arange(size, device=device, dtype=dtype) - radius
    kernel = torch.exp(-x ** 2 / (2 * sigma ** 2 + 1e-8))
    return kernel / (kernel.sum() + 1e-8)


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur."""
    if sigma < 0.3:
        return x
    kernel = _gaussian_kernel_1d(sigma, x.device, x.dtype)
    k = kernel.numel()
    pad = k // 2
    kx = kernel.view(1, 1, 1, k)
    ky = kernel.view(1, 1, k, 1)
    out = F.pad(x, (pad, pad, 0, 0), mode="reflect")
    out = F.conv2d(out, kx, groups=1)
    out = F.pad(out, (0, 0, pad, pad), mode="reflect")
    out = F.conv2d(out, ky, groups=1)
    return out


def _hessian_2d(
    x: torch.Tensor, sigma: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Hessian components at scale sigma.

    Args:
        x: (B, 1, H, W) grayscale image.
        sigma: Gaussian smoothing scale.

    Returns:
        (Ixx, Ixy, Iyy) each (B, 1, H, W), scale-normalized by σ².
    """
    blurred = _gaussian_blur(x, sigma)
    padded = F.pad(blurred, (1, 1, 1, 1), mode="reflect")
    Ixx = padded[:, :, 1:-1, 2:] - 2 * blurred + padded[:, :, 1:-1, :-2]
    Iyy = padded[:, :, 2:, 1:-1] - 2 * blurred + padded[:, :, :-2, 1:-1]
    Ixy = (
        padded[:, :, 2:, 2:]
        - padded[:, :, 2:, :-2]
        - padded[:, :, :-2, 2:]
        + padded[:, :, :-2, :-2]
    ) / 4.0
    scale = sigma ** 2
    return Ixx * scale, Ixy * scale, Iyy * scale


def _eigenvectors_2d(
    Ixx: torch.Tensor, Ixy: torch.Tensor, Iyy: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the primary eigenvector (vessel tangent) of the 2x2 Hessian.

    For tubular structures, the eigenvalue with LARGER absolute value (λ₂)
    has its eigenvector pointing PERPENDICULAR to the tube (across the
    vessel), while the eigenvalue with SMALLER absolute value (λ₁) has
    its eigenvector pointing ALONG the vessel tangent.

    We return the λ₁ eigenvector (vessel tangent direction).

    Returns:
        (cos_theta, sin_theta) each (B, 1, H, W), vessel tangent direction.
    """
    # For a 2x2 symmetric matrix [[a, b], [b, c]]:
    # eigenvalues: λ = (a+c)/2 ± sqrt(((a-c)/2)² + b²)
    # eigenvector for λ: (b, λ - a) normalized

    trace = Ixx + Iyy
    diff = Ixx - Iyy
    discriminant = torch.sqrt(diff ** 2 / 4.0 + Ixy ** 2 + 1e-8)

    # Compute both eigenvalues
    lam_plus = trace / 2.0 + discriminant
    lam_minus = trace / 2.0 - discriminant

    # λ₁ = eigenvalue with SMALLER absolute value (vessel tangent)
    use_plus_for_small = lam_plus.abs() <= lam_minus.abs()
    lam1 = torch.where(use_plus_for_small, lam_plus, lam_minus)

    # Eigenvector for λ₁: (Ixy, λ₁ - Ixx) normalized
    vx = Ixy
    vy = lam1 - Ixx

    # Normalize to unit vector
    norm = torch.sqrt(vx ** 2 + vy ** 2 + 1e-8)
    cos_theta = vx / norm
    sin_theta = vy / norm

    return cos_theta, sin_theta


def hessian_direction_field(
    image: torch.Tensor,
    sigmas: Sequence[float] = (1.0, 2.0, 4.0),
) -> torch.Tensor:
    """Compute multi-scale Hessian direction field.

    At each pixel, computes the vessel orientation from the Hessian
    eigenvector across multiple scales, taking the orientation from
    the scale with strongest vesselness response.

    Args:
        image: (C, H, W) or (B, C, H, W) input image.
        sigmas: Gaussian scales for multi-scale Hessian.

    Returns:
        (2, H, W) or (B, 2, H, W) direction field [cos θ, sin θ].
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

    # Normalize to [0, 1]
    g_min = gray.amin(dim=(-2, -1), keepdim=True)
    g_max = gray.amax(dim=(-2, -1), keepdim=True)
    gray = (gray - g_min) / (g_max - g_min + 1e-8)

    best_cos = torch.zeros_like(gray)
    best_sin = torch.zeros_like(gray)
    best_response = torch.zeros_like(gray)

    for sigma in sigmas:
        Ixx, Ixy, Iyy = _hessian_2d(gray, sigma)

        # Vesselness response (Frangi-like): strength of tubularity
        lam1 = (Ixx + Iyy) / 2.0 - torch.sqrt((Ixx - Iyy) ** 2 / 4 + Ixy ** 2 + 1e-8)
        lam2 = (Ixx + Iyy) / 2.0 + torch.sqrt((Ixx - Iyy) ** 2 / 4 + Ixy ** 2 + 1e-8)
        response = lam2.abs()  # Strong λ₂ = strong vessel evidence

        cos_t, sin_t = _eigenvectors_2d(Ixx, Ixy, Iyy)

        # Keep orientation from scale with strongest response
        update = response > best_response
        best_cos = torch.where(update, cos_t, best_cos)
        best_sin = torch.where(update, sin_t, best_sin)
        best_response = torch.where(update, response, best_response)

    # Stack to 2-channel field
    direction_field = torch.cat([best_cos, best_sin], dim=1)  # (B, 2, H, W)

    if squeeze:
        direction_field = direction_field.squeeze(0)

    return direction_field
