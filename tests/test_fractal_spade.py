"""Tests for FractalSPADE gating module."""

import torch
import pytest

from fractal_swin_unet.models.gating import FractalSPADE, apply_fractal_gate
from fractal_swin_unet.models.fractal_prior_swin_unet import FractalPriorSwinUNet


class TestFractalSPADE:
    """Unit tests for the FractalSPADE module."""

    def test_output_shape(self) -> None:
        """SPADE output matches skip shape."""
        spade = FractalSPADE(skip_channels=32, prior_channels=1, hidden_dim=16)
        skip = torch.randn(2, 32, 16, 16)
        lfd = torch.randn(2, 1, 16, 16)
        out = spade(skip, lfd)
        assert out.shape == skip.shape

    def test_spatial_resize(self) -> None:
        """SPADE handles mismatched spatial dims by resizing prior."""
        spade = FractalSPADE(skip_channels=64, prior_channels=1, hidden_dim=16)
        skip = torch.randn(1, 64, 32, 32)
        lfd = torch.randn(1, 1, 128, 128)  # different spatial size
        out = spade(skip, lfd)
        assert out.shape == skip.shape

    def test_initial_identity(self) -> None:
        """At init (gamma=0, beta=0), SPADE ≈ InstanceNorm."""
        spade = FractalSPADE(skip_channels=16, prior_channels=1, hidden_dim=8)
        skip = torch.randn(2, 16, 8, 8)
        lfd = torch.randn(2, 1, 8, 8)
        out = spade(skip, lfd)
        # Should be close to instance-normalized skip
        from torch.nn import InstanceNorm2d
        norm = InstanceNorm2d(16, affine=False)
        expected = norm(skip)
        assert torch.allclose(out, expected, atol=1e-5), \
            f"Max diff: {(out - expected).abs().max().item()}"

    def test_gradient_flow(self) -> None:
        """Gradients flow through SPADE to both skip and prior."""
        spade = FractalSPADE(skip_channels=16, prior_channels=1, hidden_dim=8)
        # Perturb gamma/beta weights away from zero so lfd path has gradient
        with torch.no_grad():
            spade.gamma_conv.weight.add_(0.01)
            spade.beta_conv.weight.add_(0.01)
        skip = torch.randn(1, 16, 8, 8, requires_grad=True)
        lfd = torch.randn(1, 1, 8, 8, requires_grad=True)
        out = spade(skip, lfd)
        loss = out.sum()
        loss.backward()
        assert skip.grad is not None and skip.grad.abs().sum() > 0
        assert lfd.grad is not None and lfd.grad.abs().sum() > 0


class TestFractalPriorSwinUNetSPADE:
    """Integration tests for SPADE in the full model."""

    def test_spade_forward(self) -> None:
        """Full model with SPADE produces correct output shape."""
        model = FractalPriorSwinUNet(
            in_channels=3,
            embed_dim=16,
            depths=(1, 1, 1),
            enable_fractal_gate=False,
            use_fractal_spade=True,
            spade_hidden=8,
        )
        x = torch.randn(1, 3, 32, 32)
        lfd = torch.randn(1, 1, 32, 32)
        out = model(x, lfd_map=lfd)
        assert out.shape == (1, 1, 32, 32)

    def test_spade_vs_gate_different_outputs(self) -> None:
        """SPADE and old gate produce different outputs (not identical)."""
        torch.manual_seed(42)
        model_gate = FractalPriorSwinUNet(
            in_channels=3, embed_dim=16, depths=(1, 1, 1),
            enable_fractal_gate=True, use_fractal_spade=False,
        )
        torch.manual_seed(42)
        model_spade = FractalPriorSwinUNet(
            in_channels=3, embed_dim=16, depths=(1, 1, 1),
            enable_fractal_gate=False, use_fractal_spade=True, spade_hidden=8,
        )
        x = torch.randn(1, 3, 32, 32)
        lfd = torch.randn(1, 1, 32, 32)
        out_gate = model_gate(x, lfd_map=lfd)
        out_spade = model_spade(x, lfd_map=lfd)
        # They share the same encoder weights (same seed) but different gating
        # so outputs should differ
        assert not torch.allclose(out_gate, out_spade, atol=1e-3)

    def test_spade_no_prior_fallback(self) -> None:
        """When no prior is provided, SPADE computes it on-the-fly."""
        model = FractalPriorSwinUNet(
            in_channels=3, embed_dim=16, depths=(1, 1, 1),
            enable_fractal_gate=False, use_fractal_spade=True,
        )
        x = torch.randn(1, 3, 32, 32)
        out = model(x)  # no lfd_map — should compute internally
        assert out.shape == (1, 1, 32, 32)

    def test_param_count_increase(self) -> None:
        """SPADE model has more learnable params than the old gate model."""
        model_gate = FractalPriorSwinUNet(
            in_channels=3, embed_dim=16, depths=(1, 1, 1),
            enable_fractal_gate=True, use_fractal_spade=False,
        )
        model_spade = FractalPriorSwinUNet(
            in_channels=3, embed_dim=16, depths=(1, 1, 1),
            enable_fractal_gate=False, use_fractal_spade=True, spade_hidden=16,
        )
        params_gate = sum(p.numel() for p in model_gate.parameters())
        params_spade = sum(p.numel() for p in model_spade.parameters())
        # SPADE should add parameters (the shared conv + gamma + beta convs)
        assert params_spade > params_gate
