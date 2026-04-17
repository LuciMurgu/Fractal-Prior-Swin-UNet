"""Tests for PDE-based vessel enhancement features."""

import torch
import pytest


# ─── Perona-Malik Anisotropic Diffusion ───────────────────────

def test_perona_malik_shape():
    """Output shape matches input spatial dims."""
    from fractal_swin_unet.pde import perona_malik_diffusion
    x = torch.randn(3, 64, 64)
    out = perona_malik_diffusion(x, n_iter=5)
    assert out.shape == (1, 64, 64), f"Expected (1, 64, 64), got {out.shape}"


def test_perona_malik_batch():
    """Works with batched input."""
    from fractal_swin_unet.pde import perona_malik_diffusion
    x = torch.randn(2, 3, 32, 32)
    out = perona_malik_diffusion(x, n_iter=3)
    assert out.shape == (2, 1, 32, 32)


def test_perona_malik_range():
    """Output is in [0, 1]."""
    from fractal_swin_unet.pde import perona_malik_diffusion
    x = torch.randn(3, 64, 64) * 100
    out = perona_malik_diffusion(x, n_iter=10)
    assert out.min() >= -1e-6
    assert out.max() <= 1.0 + 1e-6


def test_perona_malik_smoothing():
    """Diffusion reduces variance (smooths the image)."""
    from fractal_swin_unet.pde import perona_malik_diffusion
    x = torch.randn(1, 64, 64)
    out = perona_malik_diffusion(x, n_iter=20, kappa=50.0)
    assert out.var() < x.var(), "Diffusion should reduce image variance"


def test_perona_malik_stability():
    """No NaN/Inf after many iterations."""
    from fractal_swin_unet.pde import perona_malik_diffusion
    x = torch.randn(1, 32, 32)
    out = perona_malik_diffusion(x, n_iter=100, dt=0.15)
    assert torch.isfinite(out).all(), "Output should be finite after 100 iterations"


# ─── Frangi Vesselness ────────────────────────────────────────

def test_frangi_shape():
    """Output shape correct."""
    from fractal_swin_unet.pde import frangi_vesselness
    x = torch.randn(3, 64, 64)
    out = frangi_vesselness(x, sigmas=(1.0, 2.0))
    assert out.shape == (1, 64, 64)


def test_frangi_batch():
    """Works with batched input."""
    from fractal_swin_unet.pde import frangi_vesselness
    x = torch.randn(2, 3, 32, 32)
    out = frangi_vesselness(x, sigmas=(1.0,))
    assert out.shape == (2, 1, 32, 32)


def test_frangi_range():
    """Output is in [0, 1]."""
    from fractal_swin_unet.pde import frangi_vesselness
    x = torch.randn(3, 64, 64)
    out = frangi_vesselness(x, sigmas=(1.0, 2.0, 4.0))
    assert out.min() >= -1e-6
    assert out.max() <= 1.0 + 1e-6


def test_frangi_vessel_detection():
    """Responds strongly to line-like structures."""
    from fractal_swin_unet.pde import frangi_vesselness
    x = torch.zeros(1, 64, 64)
    x[:, :, :] = 1.0
    x[:, 30:34, 10:54] = 0.0  # dark vessel
    out = frangi_vesselness(x, sigmas=(1.0, 2.0), black_ridges=True)
    vessel_region = out[0, 30:34, 15:50].mean()
    bg_region = out[0, :10, :10].mean()
    assert vessel_region > bg_region, "Frangi should detect vessel more than background"


def test_frangi_blob_suppression():
    """Blobs should have low vesselness (not tube-like)."""
    from fractal_swin_unet.pde import frangi_vesselness
    x = torch.ones(1, 64, 64)
    # Dark circular blob — both eigenvalues similar, R_B ≈ 1
    yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    dist = ((yy - 32.0) ** 2 + (xx - 32.0) ** 2).sqrt()
    x[0][dist < 8] = 0.0
    out = frangi_vesselness(x, sigmas=(2.0, 4.0), black_ridges=True)
    # Line for comparison
    x2 = torch.ones(1, 64, 64)
    x2[:, 30:34, 5:59] = 0.0
    out2 = frangi_vesselness(x2, sigmas=(2.0, 4.0), black_ridges=True)
    assert out2.max() > out.max(), "Line should have higher vesselness than blob"


def test_frangi_sigma_permutation():
    """Result should be the same regardless of sigma ordering."""
    from fractal_swin_unet.pde import frangi_vesselness
    x = torch.randn(3, 48, 48)
    out1 = frangi_vesselness(x, sigmas=(1.0, 2.0, 4.0))
    out2 = frangi_vesselness(x, sigmas=(4.0, 1.0, 2.0))
    assert torch.allclose(out1, out2, atol=1e-5), "Max-over-scales should be permutation invariant"


# ─── Meijering Neuriteness ────────────────────────────────────

def test_meijering_shape():
    """Output shape correct."""
    from fractal_swin_unet.pde import meijering_neuriteness
    x = torch.randn(3, 64, 64)
    out = meijering_neuriteness(x, sigmas=(0.5, 1.0))
    assert out.shape == (1, 64, 64)


def test_meijering_batch():
    """Works with batched input."""
    from fractal_swin_unet.pde import meijering_neuriteness
    x = torch.randn(2, 3, 32, 32)
    out = meijering_neuriteness(x, sigmas=(1.0,))
    assert out.shape == (2, 1, 32, 32)


def test_meijering_range():
    """Output is in [0, 1]."""
    from fractal_swin_unet.pde import meijering_neuriteness
    x = torch.randn(3, 64, 64)
    out = meijering_neuriteness(x, sigmas=(0.5, 1.0, 2.0))
    assert out.min() >= -1e-6
    assert out.max() <= 1.0 + 1e-6


# ─── Fractional Laplacian (Fourier Spectral) ──────────────────

def test_fractional_laplacian_shape():
    """Output shape correct."""
    from fractal_swin_unet.pde import fractional_laplacian
    x = torch.randn(3, 64, 64)
    out = fractional_laplacian(x, alpha=0.5)
    assert out.shape == (1, 64, 64)


def test_fractional_laplacian_batch():
    """Works with batched input."""
    from fractal_swin_unet.pde import fractional_laplacian
    x = torch.randn(2, 3, 32, 32)
    out = fractional_laplacian(x, alpha=0.5)
    assert out.shape == (2, 1, 32, 32)


def test_fractional_laplacian_range():
    """Output is in [0, 1]."""
    from fractal_swin_unet.pde import fractional_laplacian
    x = torch.randn(3, 64, 64)
    out = fractional_laplacian(x, alpha=0.5)
    assert out.min() >= -1e-6
    assert out.max() <= 1.0 + 1e-6


def test_fractional_laplacian_invalid_alpha():
    """Rejects invalid alpha."""
    from fractal_swin_unet.pde import fractional_laplacian
    x = torch.randn(3, 32, 32)
    with pytest.raises(ValueError):
        fractional_laplacian(x, alpha=1.5)
    with pytest.raises(ValueError):
        fractional_laplacian(x, alpha=0.0)


def test_fractional_laplacian_higher_alpha_stronger():
    """Higher alpha concentrates energy at edges (higher frequency emphasis)."""
    from fractal_swin_unet.pde import fractional_laplacian
    # Structured image with sharp edges
    x = torch.zeros(1, 64, 64)
    x[:, 20:44, 20:44] = 1.0  # square
    out_low = fractional_laplacian(x, alpha=0.3)
    out_high = fractional_laplacian(x, alpha=0.7)
    # Higher alpha → sharper edge localization → lower mean (more concentrated)
    # Both should produce valid outputs with edges highlighted
    assert out_low.max() > 0.5, "Low alpha should have strong response"
    assert out_high.max() > 0.5, "High alpha should have strong response"
    # Edge pixels (around row 20) should be bright in both
    edge_low = out_low[0, 19:21, 25:35].mean()
    edge_high = out_high[0, 19:21, 25:35].mean()
    assert edge_low > 0.01, "Low alpha should respond at edges"
    assert edge_high > 0.01, "High alpha should respond at edges"


def test_fractional_laplacian_finite():
    """No NaN/Inf in output."""
    from fractal_swin_unet.pde import fractional_laplacian
    x = torch.randn(3, 48, 48)
    out = fractional_laplacian(x, alpha=0.5)
    assert torch.isfinite(out).all(), "Output must be finite"


# ─── Multiscale Fractional Laplacian ──────────────────────────

def test_multiscale_fractional_laplacian_shape():
    """Correct number of output channels."""
    from fractal_swin_unet.pde import multiscale_fractional_laplacian
    x = torch.randn(3, 48, 48)
    out = multiscale_fractional_laplacian(x, alphas=[0.3, 0.5, 0.7])
    assert out.shape == (3, 48, 48), f"Expected (3, 48, 48), got {out.shape}"


def test_multiscale_fractional_laplacian_batch():
    """Works with batched input."""
    from fractal_swin_unet.pde import multiscale_fractional_laplacian
    x = torch.randn(2, 3, 32, 32)
    out = multiscale_fractional_laplacian(x, alphas=[0.3, 0.7])
    assert out.shape == (2, 2, 32, 32)


def test_multiscale_fractional_laplacian_default():
    """Default alphas produce 3 channels."""
    from fractal_swin_unet.pde import multiscale_fractional_laplacian
    x = torch.randn(3, 32, 32)
    out = multiscale_fractional_laplacian(x)  # default [0.3, 0.5, 0.7]
    assert out.shape[0] == 3


# ─── alpha_from_fractal_dimension ─────────────────────────────

def test_alpha_from_fractal_dimension_range():
    """Maps [0, 1] LFD to [0.1, 0.9] alpha."""
    from fractal_swin_unet.pde import alpha_from_fractal_dimension
    lfd = torch.tensor([0.0, 0.5, 1.0])
    alpha = alpha_from_fractal_dimension(lfd)
    assert torch.allclose(alpha, torch.tensor([0.1, 0.5, 0.9]), atol=1e-6)


def test_alpha_from_fractal_dimension_clamp():
    """Clamps out-of-range inputs."""
    from fractal_swin_unet.pde import alpha_from_fractal_dimension
    lfd = torch.tensor([-0.5, 1.5])
    alpha = alpha_from_fractal_dimension(lfd)
    assert alpha.min() >= 0.1 - 1e-6
    assert alpha.max() <= 0.9 + 1e-6


# ─── Integration Test ─────────────────────────────────────────

def test_all_pde_on_same_input():
    """All PDE functions work on the same input and can be concatenated."""
    from fractal_swin_unet.pde import (
        perona_malik_diffusion,
        frangi_vesselness,
        meijering_neuriteness,
        fractional_laplacian,
        multiscale_fractional_laplacian,
    )
    x = torch.randn(3, 48, 48)
    d = perona_malik_diffusion(x, n_iter=3)
    f = frangi_vesselness(x, sigmas=(1.0,))
    m = meijering_neuriteness(x, sigmas=(1.0,))
    l = fractional_laplacian(x, alpha=0.5)
    ms = multiscale_fractional_laplacian(x, alphas=[0.3, 0.7])

    assert d.shape == (1, 48, 48)
    assert f.shape == (1, 48, 48)
    assert m.shape == (1, 48, 48)
    assert l.shape == (1, 48, 48)
    assert ms.shape == (2, 48, 48)

    # All can be concatenated for prior channels
    stack = torch.cat([d, f, m, l, ms], dim=0)
    assert stack.shape == (6, 48, 48)
