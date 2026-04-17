import torch

from fractal_swin_unet.fractal import LFDParams, compute_lfd_map
from fractal_swin_unet.model import SimpleSwinUNet
from fractal_swin_unet.seed import set_deterministic_seed


def test_forward_pass_shapes() -> None:
    set_deterministic_seed(123)
    x = torch.randn(2, 1, 64, 64)
    params = LFDParams(window_size=16, stride=8, box_sizes=(2, 4, 8))
    lfd = compute_lfd_map(x[0], params=params).unsqueeze(0).unsqueeze(0)

    model = SimpleSwinUNet(in_channels=1)
    y = model(x, lfd_map=lfd)

    assert y.shape == (2, 1, 64, 64)
