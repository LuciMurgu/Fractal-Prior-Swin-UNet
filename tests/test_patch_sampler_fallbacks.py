import torch

from fractal_swin_unet.data.sampling import PatchCenterSampler, PatchSamplingConfig


def test_sampler_fallback_empty_mask() -> None:
    mask = torch.zeros(16, 16)
    config = PatchSamplingConfig(enabled=True, mode="vessel_aware", p_vessel=0.9, p_background=0.1)
    sampler = PatchCenterSampler((8, 8), config)
    generator = torch.Generator().manual_seed(123)
    cached = sampler.compute_center_masks(mask)
    cy, cx = sampler.sample_center(mask, generator, cached)
    assert 0 <= cy < mask.shape[0]
    assert 0 <= cx < mask.shape[1]


def test_sampler_fallback_full_mask() -> None:
    mask = torch.ones(16, 16)
    config = PatchSamplingConfig(enabled=True, mode="vessel_aware", p_vessel=0.1, p_background=0.9)
    sampler = PatchCenterSampler((8, 8), config)
    generator = torch.Generator().manual_seed(456)
    cached = sampler.compute_center_masks(mask)
    cy, cx = sampler.sample_center(mask, generator, cached)
    assert 0 <= cy < mask.shape[0]
    assert 0 <= cx < mask.shape[1]
