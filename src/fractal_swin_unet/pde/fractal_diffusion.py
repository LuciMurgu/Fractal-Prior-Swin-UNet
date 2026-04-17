"""Fractal-Aware Anisotropic Diffusion — learnable, end-to-end differentiable.

Implements the curvature-aware anisotropic diffusion PDE:

    ∂u/∂t − α·ψ′(|∇u|·|∇²u|)·∇·(φ_fractal(‖∇u_σ‖, D)·∇u) + λ(u − u₀) = 0

where the conductance φ is modulated by the local fractal dimension map D(x):

    φ_fractal(s, x) = φ(s) · (1 − ω·D(x))

This creates a PDE preprocessing stage that is mathematically coherent
with downstream fractal priors: high-FD regions (bifurcations, capillaries)
get minimal diffusion (preserving detail), while low-FD regions (background)
get aggressive smoothing.

All PDE parameters (α, λ, σ, β, ξ, η, ω, ν, γ) are nn.Parameters and
train end-to-end via backprop through unrolled Euler iterations.

Reference: Extended Perona-Malik with curvature-sensitive modulation.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _gaussian_blur_learnable(
    x: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    """Separable Gaussian blur with differentiable sigma.

    Uses a fixed kernel size (determined by sigma value) but differentiable
    weights so gradients flow through sigma.

    Args:
        x: (B, 1, H, W) input.
        sigma: scalar tensor (learnable parameter).

    Returns:
        Blurred tensor, same shape as x.
    """
    sigma_val = sigma.detach().item()
    if sigma_val < 0.3:
        return x

    radius = max(1, int(math.ceil(3.0 * sigma_val)))
    size = 2 * radius + 1
    coords = torch.arange(size, device=x.device, dtype=x.dtype) - radius

    # Differentiable Gaussian weights (sigma is a Parameter)
    kernel = torch.exp(-coords ** 2 / (2 * sigma ** 2 + 1e-8))
    kernel = kernel / (kernel.sum() + 1e-8)

    pad = radius
    kx = kernel.view(1, 1, 1, size)
    ky = kernel.view(1, 1, size, 1)

    out = F.pad(x, (pad, pad, 0, 0), mode="reflect")
    out = F.conv2d(out, kx, groups=1)
    out = F.pad(out, (0, 0, pad, pad), mode="reflect")
    out = F.conv2d(out, ky, groups=1)
    return out


def _spatial_gradient(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Central-difference spatial gradients.

    Args:
        x: (B, 1, H, W)

    Returns:
        (grad_y, grad_x) each (B, 1, H, W).
    """
    padded = F.pad(x, (1, 1, 1, 1), mode="reflect")
    grad_y = (padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1]) / 2.0
    grad_x = (padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2]) / 2.0
    return grad_y, grad_x


def _laplacian(x: torch.Tensor) -> torch.Tensor:
    """Discrete Laplacian via 5-point stencil.

    Args:
        x: (B, 1, H, W)

    Returns:
        (B, 1, H, W) Laplacian.
    """
    padded = F.pad(x, (1, 1, 1, 1), mode="reflect")
    lap = (
        padded[:, :, :-2, 1:-1]
        + padded[:, :, 2:, 1:-1]
        + padded[:, :, 1:-1, :-2]
        + padded[:, :, 1:-1, 2:]
        - 4 * x
    )
    return lap


class FractalAnisotropicDiffusion(nn.Module):
    """Learnable fractal-weighted anisotropic diffusion.

    All PDE parameters are nn.Parameters trained end-to-end.
    The module takes an image and an LFD map, and returns:
    - The PDE-enhanced (diffused) image
    - The edge residual (original - diffused)

    These form a 3-channel prior stack [LFD, enhanced, edge_residual]
    for downstream SPADE conditioning.

    Args:
        n_steps: Number of Euler integration steps.
        dt: Time step size (fixed for stability).
    """

    def __init__(self, n_steps: int = 5, dt: float = 0.1) -> None:
        super().__init__()
        self.n_steps = n_steps
        self.dt = dt

        # --- Learnable PDE parameters (constrained via sigmoid/softplus) ---

        # Diffusion strength α ∈ [0.6, 2.0)
        self._alpha = nn.Parameter(torch.tensor(0.0))

        # Fidelity weight λ ∈ [0.01, 0.2)
        self._lambda = nn.Parameter(torch.tensor(-2.0))

        # Gaussian smoothing σ ∈ [0.5, 5.0]
        self._log_sigma = nn.Parameter(torch.tensor(0.5))

        # Conductance: φ(s) = β·√(ξ / (η·s² + ε))
        # β ∈ (1, 5], ξ ∈ (1, 5], η ∈ (0, 1], ε = small constant
        self._log_beta = nn.Parameter(torch.tensor(0.5))
        self._log_xi = nn.Parameter(torch.tensor(0.5))
        self._eta = nn.Parameter(torch.tensor(0.3))  # sigmoid → (0, 1)

        # Curvature modulation: ψ′(s) = ε_c·√(ν·s³ + γ)
        # ν ∈ (0, 1], γ ∈ (1, 4]
        self._nu = nn.Parameter(torch.tensor(0.0))   # sigmoid → (0, 1)
        self._log_gamma = nn.Parameter(torch.tensor(0.5))

        # Fractal modulation strength ω ∈ [0, 1]
        self._omega = nn.Parameter(torch.tensor(0.5))

    @property
    def alpha(self) -> torch.Tensor:
        """Diffusion strength α ∈ [0.6, 2.0)."""
        return 0.6 + 1.4 * torch.sigmoid(self._alpha)

    @property
    def lam(self) -> torch.Tensor:
        """Fidelity weight λ ∈ [0.01, 0.2)."""
        return 0.01 + 0.19 * torch.sigmoid(self._lambda)

    @property
    def sigma(self) -> torch.Tensor:
        """Gaussian smoothing σ ∈ [0.5, 5.0]."""
        return 0.5 + 4.5 * torch.sigmoid(self._log_sigma)

    @property
    def beta(self) -> torch.Tensor:
        """Conductance scale β ∈ (1, 5]."""
        return 1.0 + 4.0 * torch.sigmoid(self._log_beta)

    @property
    def xi(self) -> torch.Tensor:
        """Conductance numerator ξ ∈ (1, 5]."""
        return 1.0 + 4.0 * torch.sigmoid(self._log_xi)

    @property
    def eta(self) -> torch.Tensor:
        """Conductance denominator scale η ∈ (0, 1]."""
        return torch.sigmoid(self._eta)

    @property
    def nu(self) -> torch.Tensor:
        """Curvature power ν ∈ (0, 1]."""
        return torch.sigmoid(self._nu)

    @property
    def gamma(self) -> torch.Tensor:
        """Curvature offset γ ∈ (1, 4]."""
        return 1.0 + 3.0 * torch.sigmoid(self._log_gamma)

    @property
    def omega(self) -> torch.Tensor:
        """Fractal modulation strength ω ∈ [0, 1]."""
        return torch.sigmoid(self._omega)

    def _conductance(self, grad_mag: torch.Tensor) -> torch.Tensor:
        """Parameterized conductance: φ(s) = β·√(ξ / (η·s² + ε)).

        Heavy-tailed compared to Gaussian exp(-s²/κ²), better preserving
        weak edges (thin capillaries).
        """
        eps = 1e-6
        return self.beta * torch.sqrt(
            self.xi / (self.eta * grad_mag ** 2 + eps)
        )

    def _curvature_modulation(
        self, grad_mag: torch.Tensor, laplacian_abs: torch.Tensor
    ) -> torch.Tensor:
        """Curvature-sensitive modulation: ψ′(s) = ε_c·√(ν·s³ + γ).

        s = |∇u|·|∇²u| captures second-order geometry (curvature).
        """
        eps_c = 1e-4
        s = grad_mag * laplacian_abs
        return eps_c * torch.sqrt(self.nu * s ** 3 + self.gamma)

    def forward(
        self,
        image: torch.Tensor,
        lfd_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run fractal-weighted anisotropic diffusion.

        Args:
            image: (B, 1, H, W) grayscale image, [0, 1] normalized.
            lfd_map: (B, 1, H, W) local fractal dimension map.
                     Must be normalized to [0, 1] range (0=low complexity,
                     1=high complexity). Raw FD values in [1, 2] must be
                     rescaled: lfd_norm = (D - 1.0) to get [0, 1].

        Returns:
            enhanced: (B, 1, H, W) PDE-enhanced image.
            edge_residual: (B, 1, H, W) |image - enhanced|, normalized.
        """
        u = image.clone()
        u0 = image

        for _ in range(self.n_steps):
            # 1. Gaussian-smoothed version for robust gradient
            u_sigma = _gaussian_blur_learnable(u, self.sigma)

            # 2. Gradients and Laplacian
            gy, gx = _spatial_gradient(u)
            gy_sigma, gx_sigma = _spatial_gradient(u_sigma)
            grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
            grad_mag_sigma = torch.sqrt(gx_sigma ** 2 + gy_sigma ** 2 + 1e-8)
            lap = _laplacian(u)
            lap_abs = lap.abs()

            # 3. Conductance with fractal weighting
            phi = self._conductance(grad_mag_sigma)
            # Clamp phi to prevent instability
            phi = phi.clamp(max=10.0)
            # Fractal modulation: high FD → low conductance → preserve detail
            # Clamp to [0, 1] to prevent sign-flip if lfd_map exceeds [0,1]
            fractal_weight = (1.0 - self.omega * lfd_map).clamp(min=0.0, max=1.0)
            phi_fractal = phi * fractal_weight

            # 4. Curvature modulation
            psi_prime = self._curvature_modulation(grad_mag, lap_abs)

            # 5. Divergence: div(φ_fractal · ∇u) via directional fluxes
            padded_u = F.pad(u, (1, 1, 1, 1), mode="reflect")
            dn = padded_u[:, :, :-2, 1:-1] - u  # north
            ds = padded_u[:, :, 2:, 1:-1] - u    # south
            de = padded_u[:, :, 1:-1, 2:] - u    # east
            dw = padded_u[:, :, 1:-1, :-2] - u   # west

            padded_phi = F.pad(phi_fractal, (1, 1, 1, 1), mode="reflect")
            cn = padded_phi[:, :, :-2, 1:-1]
            cs = padded_phi[:, :, 2:, 1:-1]
            ce = padded_phi[:, :, 1:-1, 2:]
            cw = padded_phi[:, :, 1:-1, :-2]

            div_term = cn * dn + cs * ds + ce * de + cw * dw

            # 6. Full PDE update
            du = self.alpha * psi_prime * div_term - self.lam * (u - u0)
            u = u + self.dt * du

            # Clamp to valid range
            u = u.clamp(0.0, 1.0)

        # Edge residual = what the PDE removed (edges, fine detail)
        edge_residual = (image - u).abs()
        # Normalize edge residual to [0, 1]
        er_max = edge_residual.amax(dim=(-2, -1), keepdim=True)
        edge_residual = edge_residual / (er_max + 1e-8)

        return u, edge_residual

    def extra_repr(self) -> str:
        with torch.no_grad():
            return (
                f"n_steps={self.n_steps}, dt={self.dt}, "
                f"α={self.alpha.item():.3f}, λ={self.lam.item():.4f}, "
                f"σ={self.sigma.item():.2f}, ω={self.omega.item():.3f}"
            )
