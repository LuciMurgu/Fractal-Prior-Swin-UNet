import torch

from fractal_swin_unet.data.transforms import random_flip_rotate


def test_transform_alignment() -> None:
    image = torch.zeros(1, 8, 8)
    prior = torch.zeros(1, 8, 8)
    image[0, :, 3] = 1.0
    prior[0, :, 3] = 1.0

    sample = {"image": image, "prior": prior}
    generator = torch.Generator().manual_seed(123)
    out = random_flip_rotate(sample, generator)

    assert torch.allclose(out["image"], out["prior"])
