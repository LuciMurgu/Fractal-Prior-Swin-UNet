"""Test that all 9 PDE parameters receive non-zero gradients.

This verifies that the PDE module is truly trainable end-to-end:
gradients flow through the unrolled Euler iterations, through the
differentiable Gaussian blur (fixed-radius kernel), through the
conductance and curvature functions, all the way back to the 9
learnable parameters.
"""

import torch
import pytest

from fractal_swin_unet.pde.fractal_diffusion import FractalAnisotropicDiffusion


class TestPDEGradientFlow:

    def test_all_9_params_receive_gradients(self):
        """Forward + backward on a dummy loss → all 9 PDE params have grad != 0."""
        pde = FractalAnisotropicDiffusion(n_steps=3, sigma_max=5.0)

        # Dummy inputs
        image = torch.rand(1, 1, 32, 32, requires_grad=False)
        lfd = torch.rand(1, 1, 32, 32, requires_grad=False)

        enhanced, edge_residual = pde(image, lfd)

        # Dummy loss: sum of outputs → ensures gradient signal
        loss = enhanced.sum() + edge_residual.sum()
        loss.backward()

        # All 9 raw parameters
        param_names = [
            "_alpha", "_lambda", "_log_sigma",
            "_log_beta", "_log_xi", "_eta",
            "_nu", "_log_gamma", "_omega",
        ]

        for name in param_names:
            param = getattr(pde, name)
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert param.grad.abs().item() > 0, (
                f"Parameter {name} has zero gradient"
            )

    def test_gradient_not_nan(self):
        """No NaN gradients in any PDE parameter."""
        pde = FractalAnisotropicDiffusion(n_steps=5, sigma_max=5.0)
        image = torch.rand(1, 1, 32, 32)
        lfd = torch.rand(1, 1, 32, 32)

        enhanced, edge = pde(image, lfd)
        loss = enhanced.mean() + edge.mean()
        loss.backward()

        for name, param in pde.named_parameters():
            assert param.grad is not None, f"{name} has no grad"
            assert not torch.isnan(param.grad).any(), f"{name} has NaN grad"
            assert torch.isfinite(param.grad).all(), f"{name} has Inf grad"

    def test_get_param_dict_returns_all_9(self):
        """get_param_dict() returns exactly 9 named parameters."""
        pde = FractalAnisotropicDiffusion(n_steps=3)
        d = pde.get_param_dict()

        expected_keys = {"alpha", "lambda", "sigma", "beta", "xi", "eta", "nu", "gamma", "omega"}
        assert set(d.keys()) == expected_keys, f"Missing keys: {expected_keys - set(d.keys())}"
        for k, v in d.items():
            assert isinstance(v, float), f"{k} should be float, got {type(v)}"

    def test_clip_pde_gradients(self):
        """clip_pde_gradients() reduces gradient norms below max_norm."""
        pde = FractalAnisotropicDiffusion(n_steps=2, sigma_max=3.0)
        image = torch.rand(1, 1, 16, 16)
        lfd = torch.rand(1, 1, 16, 16)

        enhanced, edge = pde(image, lfd)
        # Amplified loss to create large gradients
        loss = 1000.0 * (enhanced.sum() + edge.sum())
        loss.backward()

        # Check some grads are large before clipping
        grad_norms_before = [p.grad.norm().item() for p in pde.parameters() if p.grad is not None]

        pde.clip_pde_gradients(max_norm=0.5)

        # After clipping, total norm should be ≤ max_norm
        all_grads = [p.grad for p in pde.parameters() if p.grad is not None]
        total_norm = torch.sqrt(sum(g.norm() ** 2 for g in all_grads)).item()
        assert total_norm <= 0.5 + 1e-4, f"Clipped norm {total_norm} > 0.5"

    def test_sigma_max_sets_fixed_radius(self):
        """sigma_max correctly determines the fixed blur kernel radius."""
        import math
        pde = FractalAnisotropicDiffusion(n_steps=2, sigma_max=4.0)
        expected_radius = int(math.ceil(2.5 * 4.0))  # 10
        assert pde._blur_radius == expected_radius

    def test_n_steps_configurable(self):
        """n_steps is correctly set from constructor."""
        pde3 = FractalAnisotropicDiffusion(n_steps=3)
        pde15 = FractalAnisotropicDiffusion(n_steps=15)
        assert pde3.n_steps == 3
        assert pde15.n_steps == 15
