import torch

from fractal_swin_unet.models.gating import apply_fractal_gate


def test_gating_broadcast_shape_and_range() -> None:
    skip = torch.ones(2, 4, 8, 8)
    lfd = torch.linspace(0.0, 1.0, steps=64).reshape(1, 1, 8, 8).repeat(2, 1, 1, 1)

    gated = apply_fractal_gate(skip, lfd, alpha=1.5)

    assert gated.shape == skip.shape
    assert torch.all(gated >= 0.0)
    assert torch.all(gated <= 1.0)
