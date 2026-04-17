import torch

from fractal_swin_unet.data import InfiniteRandomPatchDataset
from fractal_swin_unet.fractal.provider import FractalPriorConfig, FractalPriorProvider
from fractal_swin_unet.models import FractalPriorSwinUNet


def test_fractal_provider_multiscale_patch_shape() -> None:
    image = torch.rand(3, 64, 64)
    patch = image[:, 8:40, 8:40]
    provider = FractalPriorProvider(
        FractalPriorConfig(
            enabled=True,
            precompute_enabled=False,
            caching_enabled=False,
            multi_scale_enabled=True,
            multi_scale_factors=(1, 2, 4),
        )
    )
    prior = provider.get_patch("sample", patch)
    assert prior.shape == (3, 32, 32)


def test_dataset_returns_multiscale_prior_channels() -> None:
    image = torch.rand(3, 64, 64)
    mask = (torch.rand(1, 64, 64) > 0.5).float()
    dataset = InfiniteRandomPatchDataset(
        image=image,
        mask=mask,
        patch_size=32,
        seed=5,
        return_fractal_prior=True,
        fractal_prior_config={
            "enabled": True,
            "multi_scale": {"enabled": True, "factors": [1, 2]},
            "caching": {"enabled": False, "mode": "off"},
            "precompute": {"enabled": False},
        },
    )
    sample = next(iter(dataset))
    assert sample["prior"].shape == (2, 32, 32)


def test_model_accepts_multichannel_prior_with_fusion() -> None:
    model = FractalPriorSwinUNet(
        in_channels=3,
        embed_dim=16,
        depths=(1, 1, 1),
        enable_fractal_gate=True,
        enable_prior_fusion=True,
    )
    x = torch.randn(2, 3, 64, 64)
    lfd_map = torch.rand(2, 3, 64, 64)
    y = model(x, lfd_map=lfd_map, expect_prior=True)
    assert y.shape == (2, 1, 64, 64)
