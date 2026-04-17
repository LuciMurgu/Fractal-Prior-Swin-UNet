import torch

from fractal_swin_unet.data import InfiniteRandomPatchDataset


def test_dataset_returns_prior() -> None:
    image = torch.rand(3, 64, 64)
    mask = (torch.rand(1, 64, 64) > 0.5).float()

    dataset = InfiniteRandomPatchDataset(
        image=image,
        mask=mask,
        patch_size=32,
        seed=1,
        return_fractal_prior=True,
        fractal_prior_config={
            "enabled": True,
            "lfd": {"window": 7, "stride": 2, "box_sizes": [2, 4]},
            "caching": {"enabled": False, "mode": "off"},
            "precompute": {"enabled": False},
        },
    )

    sample = next(iter(dataset))
    assert "prior" in sample
    assert sample["prior"].shape[-2:] == sample["image"].shape[-2:]
