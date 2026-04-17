import torch

from fractal_swin_unet.data import InfiniteRandomPatchDataset


def test_patch_sampler_bias_increases_vessel_hits() -> None:
    image = torch.zeros(1, 128, 128)
    mask = torch.zeros(1, 128, 128)
    for i in range(128):
        mask[0, i, i] = 1.0

    uniform_ds = InfiniteRandomPatchDataset(image=image, mask=mask, patch_size=16, seed=123)
    vessel_ds = InfiniteRandomPatchDataset(
        image=image,
        mask=mask,
        patch_size=16,
        seed=123,
        sampling_config={
            "enabled": True,
            "mode": "vessel_aware",
            "p_vessel": 0.9,
            "p_background": 0.1,
            "vessel_buffer": 2,
            "background_buffer": 0,
            "max_retries": 10,
        },
        sample_id="diag",
    )

    def vessel_hit_fraction(ds, n=200):
        hits = 0
        it = iter(ds)
        for _ in range(n):
            batch = next(it)
            if batch["mask"].sum().item() > 0:
                hits += 1
        return hits / n

    uniform_hits = vessel_hit_fraction(uniform_ds)
    vessel_hits = vessel_hit_fraction(vessel_ds)

    assert vessel_hits >= uniform_hits + 0.2
    assert vessel_hits >= 0.5
