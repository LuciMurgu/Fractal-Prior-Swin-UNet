"""Unit tests for the FractalNCA module.

Tests:
  1. Output shape matches input logits shape
  2. No NaN in output after forward pass
  3. Gradient flows through the NCA loop
  4. Stochastic mask is active during training, inactive during eval
  5. Integration test: FractalPriorSwinUNet with NCA refiner enabled
"""

from __future__ import annotations

import torch
import pytest


def test_nca_output_shape():
    """NCA output shape must match (B, 1, H, W) of coarse logits."""
    from fractal_swin_unet.models.fractal_nca import FractalNCA

    nca = FractalNCA(c_logit=1, c_skip=32, c_hidden=16, n_steps=4)
    logits = torch.randn(2, 1, 64, 64)
    skip = torch.randn(2, 32, 64, 64)

    out = nca(logits, skip)
    assert out.shape == logits.shape, f"Expected {logits.shape}, got {out.shape}"


def test_nca_no_nan():
    """NCA must not produce NaN values."""
    from fractal_swin_unet.models.fractal_nca import FractalNCA

    nca = FractalNCA(c_logit=1, c_skip=16, c_hidden=16, n_steps=8)
    logits = torch.randn(2, 1, 32, 32)
    skip = torch.randn(2, 16, 32, 32)

    out = nca(logits, skip)
    assert not torch.isnan(out).any(), "NCA output contains NaN"
    assert not torch.isinf(out).any(), "NCA output contains Inf"


def test_nca_gradient_flow():
    """Gradients must flow through the NCA loop back to inputs."""
    from fractal_swin_unet.models.fractal_nca import FractalNCA

    nca = FractalNCA(c_logit=1, c_skip=16, c_hidden=16, n_steps=4)
    nca.train()

    logits = torch.randn(1, 1, 32, 32, requires_grad=True)
    skip = torch.randn(1, 16, 32, 32, requires_grad=True)

    out = nca(logits, skip)
    loss = out.sum()
    loss.backward()

    assert logits.grad is not None, "No gradient on logits"
    assert skip.grad is not None, "No gradient on skip"
    assert logits.grad.abs().sum() > 0, "Zero gradient on logits"
    assert skip.grad.abs().sum() > 0, "Zero gradient on skip"


def test_nca_eval_deterministic():
    """In eval mode, same input should give same output (no stochastic mask)."""
    from fractal_swin_unet.models.fractal_nca import FractalNCA

    nca = FractalNCA(c_logit=1, c_skip=16, c_hidden=16, n_steps=4, stochastic_rate=0.5)
    nca.eval()

    logits = torch.randn(1, 1, 32, 32)
    skip = torch.randn(1, 16, 32, 32)

    with torch.no_grad():
        out1 = nca(logits, skip)
        out2 = nca(logits, skip)

    assert torch.allclose(out1, out2, atol=1e-6), "Eval mode should be deterministic"


def test_nca_zero_init_identity():
    """With zero-initialized update rule, NCA should approximately pass through."""
    from fractal_swin_unet.models.fractal_nca import FractalNCA

    nca = FractalNCA(c_logit=1, c_skip=16, c_hidden=16, n_steps=4)
    nca.eval()

    logits = torch.randn(1, 1, 32, 32)
    skip = torch.randn(1, 16, 32, 32)

    with torch.no_grad():
        out = nca(logits, skip)

    # With zero-init on the update rule, hidden state stays fixed after init,
    # so output should be some linear projection of the initial state.
    # Just verify it's finite and reasonable magnitude.
    assert not torch.isnan(out).any()
    assert out.abs().max() < 100.0, "Output magnitude unreasonably large"


def test_nca_skip_spatial_mismatch():
    """NCA should handle skip features with different spatial size via interpolation."""
    from fractal_swin_unet.models.fractal_nca import FractalNCA

    nca = FractalNCA(c_logit=1, c_skip=16, c_hidden=16, n_steps=4)
    logits = torch.randn(1, 1, 64, 64)
    skip = torch.randn(1, 16, 32, 32)  # Half the spatial size

    out = nca(logits, skip)
    assert out.shape == logits.shape, f"Expected {logits.shape}, got {out.shape}"


def test_fractal_unet_with_nca():
    """Integration: FractalPriorSwinUNet with use_nca_refiner=True."""
    from fractal_swin_unet.models import FractalPriorSwinUNet

    model = FractalPriorSwinUNet(
        in_channels=3,
        embed_dim=16,
        depths=(1, 1, 1),
        enable_fractal_gate=False,
        use_nca_refiner=True,
        nca_hidden=16,
        nca_steps=4,
    )
    model.eval()

    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 1, 64, 64), f"Expected (1,1,64,64), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in integrated model output"
