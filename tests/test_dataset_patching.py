import torch

from fractal_swin_unet.data import InfiniteRandomPatchDataset
from fractal_swin_unet.seed import set_deterministic_seed


def test_dataset_patching_shapes_and_determinism() -> None:
    set_deterministic_seed(11)
    image = torch.randn(3, 64, 64)
    mask = (torch.randn(1, 64, 64) > 0).float()

    ds1 = InfiniteRandomPatchDataset(image=image, mask=mask, patch_size=32, seed=99)
    ds2 = InfiniteRandomPatchDataset(image=image, mask=mask, patch_size=32, seed=99)

    sample1 = next(iter(ds1))
    sample2 = next(iter(ds2))

    assert sample1["image"].shape == (3, 32, 32)
    assert sample1["mask"].shape == (1, 32, 32)
    assert torch.allclose(sample1["image"], sample2["image"])
    assert torch.allclose(sample1["mask"], sample2["mask"])
