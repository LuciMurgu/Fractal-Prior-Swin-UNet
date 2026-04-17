import torch

from fractal_swin_unet.fractal import LFDParams, compute_lfd_map
from fractal_swin_unet.seed import set_deterministic_seed


def test_lfd_map_range() -> None:
    set_deterministic_seed(7)
    x = torch.randn(3, 32, 32)
    params = LFDParams(window_size=16, stride=8, box_sizes=(2, 4, 8))
    lfd = compute_lfd_map(x, params=params)

    assert lfd.shape == (32, 32)
    assert torch.all(lfd >= 0.0)
    assert torch.all(lfd <= 1.0)
