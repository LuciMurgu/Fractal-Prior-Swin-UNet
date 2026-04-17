"""Perona-Malik anisotropic diffusion (edge-preserving PDE smoothing).

Implements the PDE:
    ∂I/∂t = div(c(|∇I|) ∇I)

where c(s) = exp(-(s/κ)²) is the edge-stopping diffusivity.

This smooths homogeneous regions while preserving vessel boundaries,
acting as a connectivity-preserving denoiser.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _gradient_2d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute spatial gradients using central differences.
    
    Args:
        x: (B, 1, H, W) tensor.
    
    Returns:
        (grad_y, grad_x) each of shape (B, 1, H, W).
    """
    # Pad for central differences
    padded = F.pad(x, (1, 1, 1, 1), mode="reflect")
    grad_y = (padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1]) / 2.0
    grad_x = (padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2]) / 2.0
    return grad_y, grad_x


def perona_malik_diffusion(
    image: torch.Tensor,
    kappa: float = 30.0,
    n_iter: int = 15,
    dt: float = 0.15,
) -> torch.Tensor:
    """Apply Perona-Malik anisotropic diffusion.

    Args:
        image: Input image tensor of shape (C, H, W) or (B, C, H, W).
            Internally converted to grayscale if multi-channel.
        kappa: Edge sensitivity parameter. Higher = more smoothing.
        n_iter: Number of diffusion iterations.
        dt: Time step size (must be < 0.25 for stability).

    Returns:
        Diffused image of shape (1, H, W) or (B, 1, H, W), normalized to [0, 1].
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
    vmin = gray.amin(dim=(-2, -1), keepdim=True)
    vmax = gray.amax(dim=(-2, -1), keepdim=True)
    u = (gray - vmin) / (vmax - vmin + 1e-8)

    for _ in range(n_iter):
        gy, gx = _gradient_2d(u)
        grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

        # Edge-stopping function: c(s) = exp(-(s/κ)²)
        c = torch.exp(-(grad_mag / kappa) ** 2)

        # Divergence: div(c · ∇u) via finite differences
        # North, South, East, West flux
        padded = F.pad(u, (1, 1, 1, 1), mode="reflect")
        dn = padded[:, :, :-2, 1:-1] - u  # north
        ds = padded[:, :, 2:, 1:-1] - u   # south
        de = padded[:, :, 1:-1, 2:] - u   # east
        dw = padded[:, :, 1:-1, :-2] - u  # west

        # Conductance at each direction
        c_pad = F.pad(c, (1, 1, 1, 1), mode="reflect")
        cn = c_pad[:, :, :-2, 1:-1]
        cs = c_pad[:, :, 2:, 1:-1]
        ce = c_pad[:, :, 1:-1, 2:]
        cw = c_pad[:, :, 1:-1, :-2]

        u = u + dt * (cn * dn + cs * ds + ce * de + cw * dw)

    # Normalize output
    vmin = u.amin(dim=(-2, -1), keepdim=True)
    vmax = u.amax(dim=(-2, -1), keepdim=True)
    u = (u - vmin) / (vmax - vmin + 1e-8)

    if squeeze:
        u = u.squeeze(0)
    return u
