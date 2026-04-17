import torch

from fractal_swin_unet.data.sampling import PatchCenterSampler, PatchSamplingConfig


def test_patch_sampler_centers_in_bounds() -> None:
    mask = torch.zeros(32, 48)
    config = PatchSamplingConfig(enabled=True, mode="vessel_aware", vessel_buffer=1, background_buffer=0)
    sampler = PatchCenterSampler((15, 17), config)
    generator = torch.Generator().manual_seed(7)
    cached = sampler.compute_center_masks(mask)

    for _ in range(100):
        cy, cx = sampler.sample_center(mask, generator, cached)
        top = cy - sampler.patch_h // 2
        left = cx - sampler.patch_w // 2
        assert top >= 0
        assert left >= 0
        assert top + sampler.patch_h <= mask.shape[0]
        assert left + sampler.patch_w <= mask.shape[1]
