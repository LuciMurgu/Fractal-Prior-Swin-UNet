"""Fractional Laplacian via Fourier spectral method.

Computes (-Δ)^α I for fractional order α ∈ (0, 1) using the identity:

    F[(-Δ)^α u](ξ) = |ξ|^{2α} · F[u](ξ)

where F is the 2D Fourier transform. This is mathematically exact in the
frequency domain and connects fractal geometry to PDE theory:

    - The fractal dimension D_f of a surface relates to the Hölder
      regularity, which determines how (-Δ)^α acts on the signal.
    - Higher D_f → more energy in high frequencies → stronger fractional
      Laplacian response.
    - α close to 0: coarse structure detection (major vessels)
    - α close to 1: fine structure detection (capillaries, approaches standard Laplacian)

Reference: Fractional Sobolev spaces and PDEs on fractal domains
"""

from __future__ import annotations

import torch


def fractional_laplacian(
    image: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Compute the fractional Laplacian (-Δ)^α via Fourier spectral method.

    Args:
        image: Input tensor of shape (C, H, W) or (B, C, H, W).
        alpha: Fractional order in (0, 1).
            - 0.3: coarse structure (major vessels, bifurcations)
            - 0.5: balanced multi-scale (default, half-Laplacian)
            - 0.7: fine structure (capillaries, thin branches)

    Returns:
        Fractional Laplacian magnitude, shape (1, H, W) or (B, 1, H, W),
        normalized to [0, 1].
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

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

    # Normalize input to [0, 1]
    vmin = gray.amin(dim=(-2, -1), keepdim=True)
    vmax = gray.amax(dim=(-2, -1), keepdim=True)
    gray = (gray - vmin) / (vmax - vmin + 1e-8)

    B, _, H, W = gray.shape

    # 1. Compute 2D FFT
    F_u = torch.fft.rfft2(gray)

    # 2. Build frequency grid |ξ|^{2α}
    freq_y = torch.fft.fftfreq(H, device=gray.device, dtype=gray.dtype)  # (H,)
    freq_x = torch.fft.rfftfreq(W, device=gray.device, dtype=gray.dtype)  # (W//2+1,)
    freq_squared = freq_y[:, None] ** 2 + freq_x[None, :] ** 2  # (H, W//2+1)
    multiplier = freq_squared ** alpha  # |ξ|^{2α}

    # 3. Zero the DC component (avoid shifting the mean)
    multiplier[0, 0] = 0.0

    # 4. Apply in frequency domain
    F_result = F_u * multiplier.unsqueeze(0).unsqueeze(0)  # (B, 1, H, W//2+1)

    # 5. Inverse FFT
    result = torch.fft.irfft2(F_result, s=(H, W))

    # 6. Take absolute value (response can be negative)
    result = torch.abs(result)

    # 7. Normalize to [0, 1] per sample
    vmax = result.amax(dim=(-2, -1), keepdim=True)
    result = result / (vmax + 1e-8)

    if squeeze:
        result = result.squeeze(0)
    return result


def multiscale_fractional_laplacian(
    image: torch.Tensor,
    alphas: list[float] | None = None,
) -> torch.Tensor:
    """Compute fractional Laplacian at multiple orders as separate channels.

    This provides a scale-decomposition analogous to multi-scale Frangi but
    grounded in fractional calculus. Each α captures a different structural
    scale of the vessel tree.

    Args:
        image: Input tensor of shape (C, H, W) or (B, C, H, W).
        alphas: List of fractional orders. Default [0.3, 0.5, 0.7] gives
            coarse / balanced / fine decomposition.

    Returns:
        Multi-channel output of shape (len(alphas), H, W) or
        (B, len(alphas), H, W), each channel normalized to [0, 1].
    """
    if alphas is None:
        alphas = [0.3, 0.5, 0.7]

    squeeze = False
    if image.ndim == 3:
        image = image.unsqueeze(0)
        squeeze = True

    channels = []
    for a in alphas:
        ch = fractional_laplacian(image, alpha=a)  # (B, 1, H, W)
        channels.append(ch)

    result = torch.cat(channels, dim=1)  # (B, len(alphas), H, W)

    if squeeze:
        result = result.squeeze(0)
    return result


def alpha_from_fractal_dimension(lfd_map: torch.Tensor) -> torch.Tensor:
    """Convert local fractal dimension map to local fractional order α.

    The theoretical bridge: for a 2D image, the fractal dimension D_f
    relates to Hölder regularity s via D_f = 2 - s + 1 = 3 - s.
    The fractional Laplacian order that "resonates" with structures
    of dimension D_f is approximately α = D_f - 1.

    In practice, the LFD map is already normalized to [0, 1] from the
    DBC computation. We map this linearly to α ∈ [0.1, 0.9]:
    - Low LFD (smooth background) → low α (weak enhancement)
    - High LFD (complex vessels/bifurcations) → high α (strong enhancement)

    This creates a spatially-varying fractional order that adapts to
    local vessel complexity, providing stronger edge enhancement at
    bifurcations and weaker smoothing in homogeneous regions.

    Note: Currently for theoretical discussion in the paper. The training
    pipeline uses fixed α values for computational efficiency.

    Args:
        lfd_map: Normalized local fractal dimension map, shape (H, W) or
            (1, 1, H, W), values in [0, 1].

    Returns:
        alpha_map: Spatially-varying α values in [0.1, 0.9], same shape.
    """
    return 0.1 + 0.8 * lfd_map.clamp(0.0, 1.0)
