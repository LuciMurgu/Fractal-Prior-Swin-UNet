import torch

from fractal_swin_unet.models import SwinUNetTiny
from fractal_swin_unet.seed import set_deterministic_seed


def test_swin_unet_tiny_output_shape() -> None:
    set_deterministic_seed(101)
    model = SwinUNetTiny(in_channels=3, embed_dim=16, depths=(1, 1, 1))
    x = torch.randn(2, 3, 64, 64)
    y = model(x)

    assert y.shape == (2, 1, 64, 64)
