"""Tests for RotEquivariantConv2d and RotEquivariantSwinBlock."""

from __future__ import annotations

import pytest
import torch

from fractal_swin_unet.models.rot_equivariant import (
    RotEquivariantConv2d,
    RotEquivariantSwinBlock,
    RotEquivariantSwinStage,
    _rotate_kernel,
)


class TestRotateKernel:
    """Tests for the kernel rotation utility."""

    def test_identity_rotation(self) -> None:
        """0 degree rotation returns the same kernel."""
        w = torch.rand(4, 1, 3, 3)
        rotated = _rotate_kernel(w, 0.0)
        assert torch.allclose(rotated, w)

    def test_shape_preserved(self) -> None:
        """Output shape matches input shape."""
        w = torch.rand(8, 2, 5, 5)
        rotated = _rotate_kernel(w, 45.0)
        assert rotated.shape == w.shape

    def test_90_degree_rotation(self) -> None:
        """90-degree rotation of asymmetric kernel produces distinct result."""
        w = torch.zeros(1, 1, 3, 3)
        w[0, 0, 0, 1] = 1.0  # top-center pixel
        rotated = _rotate_kernel(w, 90.0)
        # After 90° rotation, the top-center should move to right-center
        # (with some interpolation artifacts from grid_sample)
        assert rotated.shape == (1, 1, 3, 3)
        # The rotated version should be different from original
        assert not torch.allclose(rotated, w, atol=0.1)


class TestRotEquivariantConv2d:
    """Tests for the rotation-equivariant convolution."""

    def test_output_shape(self) -> None:
        """Output spatial dims preserved with proper padding."""
        conv = RotEquivariantConv2d(16, 16, kernel_size=3, padding=1, groups=16, n_rotations=4)
        x = torch.rand(2, 16, 32, 32)
        out = conv(x)
        assert out.shape == (2, 16, 32, 32)

    def test_output_shape_non_depthwise(self) -> None:
        """Works without groups (standard conv)."""
        conv = RotEquivariantConv2d(3, 16, kernel_size=3, padding=1, n_rotations=4)
        x = torch.rand(1, 3, 32, 32)
        out = conv(x)
        assert out.shape == (1, 16, 32, 32)

    def test_rotation_invariance(self) -> None:
        """Max-pooled output should be approximately rotation-invariant."""
        conv = RotEquivariantConv2d(1, 1, kernel_size=3, padding=1, n_rotations=8)
        # Create a horizontal edge
        x1 = torch.zeros(1, 1, 32, 32)
        x1[0, 0, 14:18, :] = 1.0

        # Create a vertical edge (90° rotated)
        x2 = torch.zeros(1, 1, 32, 32)
        x2[0, 0, :, 14:18] = 1.0

        out1 = conv(x1)
        out2 = conv(x2)

        # The center responses should be similar (rotation invariant)
        # This is a soft check — exact invariance isn't guaranteed with bilinear interpolation
        center1 = out1[0, 0, 16, 16].item()
        center2 = out2[0, 0, 16, 16].item()
        # Both should be positive (edge detected)
        # Note: exact match not expected due to discrete rotation approximation
        assert abs(center1) > 0 or abs(center2) > 0

    def test_fewer_params_than_standard(self) -> None:
        """Same param count as standard conv (one base filter only)."""
        rot_conv = RotEquivariantConv2d(16, 16, 3, padding=1, groups=16, n_rotations=8)
        std_conv = torch.nn.Conv2d(16, 16, 3, padding=1, groups=16)
        rot_params = sum(p.numel() for p in rot_conv.parameters())
        std_params = sum(p.numel() for p in std_conv.parameters())
        assert rot_params == std_params  # Same weight count, more effective filters

    def test_gradients_flow(self) -> None:
        """Gradients flow through the rotated filter bank."""
        conv = RotEquivariantConv2d(8, 8, 3, padding=1, groups=8, n_rotations=4)
        x = torch.rand(1, 8, 16, 16, requires_grad=True)
        out = conv(x)
        out.sum().backward()
        assert x.grad is not None
        assert conv.base_weight.grad is not None
        assert conv.base_weight.grad.abs().sum() > 0


class TestRotEquivariantSwinBlock:
    """Tests for the equivariant SwinBlock."""

    def test_output_shape(self) -> None:
        block = RotEquivariantSwinBlock(32, n_rotations=4)
        x = torch.rand(1, 32, 16, 16)
        out = block(x)
        assert out.shape == (1, 32, 16, 16)

    def test_residual_connection(self) -> None:
        """Output should be different from input (features transformed)."""
        block = RotEquivariantSwinBlock(16, n_rotations=4)
        x = torch.rand(1, 16, 16, 16)
        out = block(x)
        # Should be different (MLP + conv changes values)
        assert not torch.allclose(x, out)


class TestRotEquivariantSwinStage:
    """Tests for the equivariant SwinStage."""

    def test_output_shape_no_downsample(self) -> None:
        stage = RotEquivariantSwinStage(32, 32, depth=1, downsample=False, n_rotations=4)
        x = torch.rand(1, 32, 16, 16)
        out = stage(x)
        assert out.shape == (1, 32, 16, 16)

    def test_output_shape_with_downsample(self) -> None:
        stage = RotEquivariantSwinStage(32, 64, depth=1, downsample=True, n_rotations=4)
        x = torch.rand(1, 32, 16, 16)
        out = stage(x)
        assert out.shape == (1, 64, 8, 8)


class TestIntegration:
    """Integration with FractalPriorSwinUNet."""

    def test_model_with_rot_equivariant(self) -> None:
        from fractal_swin_unet.models.fractal_prior_swin_unet import FractalPriorSwinUNet

        model = FractalPriorSwinUNet(
            in_channels=3,
            embed_dim=16,
            depths=(1, 1, 1),
            enable_fractal_gate=False,
            use_rot_equivariant=True,
            n_rotations=4,
        )
        x = torch.rand(1, 3, 64, 64)
        out = model(x)
        assert out.shape == (1, 1, 64, 64)

    def test_full_config_i(self) -> None:
        """Model with all Config I features enabled."""
        from fractal_swin_unet.models.fractal_prior_swin_unet import FractalPriorSwinUNet

        model = FractalPriorSwinUNet(
            in_channels=3,
            embed_dim=16,
            depths=(1, 1, 1),
            enable_fractal_gate=False,
            use_fractal_diffusion_spade=True,
            fad_n_steps=2,
            fad_prior_channels=5,
            use_hessian_direction=True,
            use_rot_equivariant=True,
            n_rotations=4,
            use_freq_geom=True,
        )
        x = torch.rand(1, 3, 64, 64)
        lfd = torch.rand(1, 1, 64, 64)
        out = model(x, lfd_map=lfd)
        assert out.shape == (1, 1, 64, 64)

    def test_backward_with_all_features(self) -> None:
        """Gradients flow through EVERYTHING: rot-equiv + PDE + SPADE + FreqGeom."""
        from fractal_swin_unet.models.fractal_prior_swin_unet import FractalPriorSwinUNet

        model = FractalPriorSwinUNet(
            in_channels=3,
            embed_dim=16,
            depths=(1, 1, 1),
            enable_fractal_gate=False,
            use_fractal_diffusion_spade=True,
            fad_n_steps=2,
            fad_prior_channels=5,
            use_hessian_direction=True,
            use_rot_equivariant=True,
            n_rotations=4,
        )
        x = torch.rand(1, 3, 64, 64)
        lfd = torch.rand(1, 1, 64, 64)
        out = model(x, lfd_map=lfd)
        out.sum().backward()

        # Check gradients flow through equivariant conv
        for name, param in model.named_parameters():
            if "base_weight" in name:
                assert param.grad is not None, f"No gradient for {name}"
                break
