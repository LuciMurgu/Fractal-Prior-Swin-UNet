import torch

from fractal_swin_unet.fractal.provider import FractalPriorConfig, FractalPriorProvider


def test_precompute_crop_equivalence() -> None:
    torch.manual_seed(0)
    image = torch.rand(3, 64, 64)
    provider = FractalPriorProvider(
        FractalPriorConfig(enabled=True, precompute_enabled=True, caching_enabled=False)
    )

    full = provider.get_full("s1", image)
    patch_box = (12, 12, 40, 40)
    patch = image[:, 12:52, 12:52]
    prior_crop = provider.get_patch("s1", patch, image_full=image, patch_box=patch_box)

    provider_no_pre = FractalPriorProvider(
        FractalPriorConfig(enabled=True, precompute_enabled=False, caching_enabled=False)
    )
    prior_direct = provider_no_pre.get_patch("s1", patch)

    assert prior_crop.shape == prior_direct.shape
    assert 0.0 <= prior_crop.min().item() <= 1.0
    assert 0.0 <= prior_direct.min().item() <= 1.0

    corr = torch.corrcoef(torch.stack([prior_crop.flatten(), prior_direct.flatten()]))[0, 1]
    assert corr > 0.7
