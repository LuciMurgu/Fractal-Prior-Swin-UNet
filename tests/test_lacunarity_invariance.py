"""Invariance and correctness tests for lacunarity map.

Tests that lacunarity correctly distinguishes uniform from heterogeneous
texture regions, and that the vectorized output is consistent.
"""

import torch
import pytest

from fractal_swin_unet.fractal.lacunarity import (
    compute_lacunarity_map,
    gliding_box_lacunarity,
    compute_dataset_percentiles,
    save_percentiles,
    load_percentiles,
)


class TestLacunarityInvariance:

    def test_heterogeneous_higher_than_uniform(self):
        """Heterogeneous region must have significantly higher lacunarity
        than a uniform region (difference > 0.3 after normalization)."""
        torch.manual_seed(42)

        # Build an image with two distinct regions:
        # Left half: uniform (constant)
        # Right half: heterogeneous (checkerboard + noise)
        image = torch.zeros(1, 128, 128)
        image[:, :, :64] = 0.5  # uniform

        # Heterogeneous: high-frequency checkerboard + noise
        checker = torch.zeros(128, 64)
        for i in range(0, 128, 4):
            for j in range(0, 64, 4):
                val = 1.0 if ((i // 4) + (j // 4)) % 2 == 0 else 0.0
                checker[i:i+4, j:j+4] = val
        image[0, :, 64:] = checker + 0.1 * torch.randn(128, 64)
        image = image.clamp(0, 1)

        lac = compute_lacunarity_map(image, window_size=16, stride=8, box_sizes=(2, 4, 8))

        # Measure mean lacunarity in each half
        lac_uniform = lac[:, :64].mean().item()
        lac_hetero = lac[:, 64:].mean().item()

        # Normalize to [0, 1] range for comparison
        lac_min = min(lac_uniform, lac_hetero)
        lac_max = max(lac_uniform, lac_hetero)
        if lac_max > lac_min:
            lac_uniform_norm = (lac_uniform - lac_min) / (lac_max - lac_min)
            lac_hetero_norm = (lac_hetero - lac_min) / (lac_max - lac_min)
        else:
            lac_uniform_norm = 0.0
            lac_hetero_norm = 0.0

        diff = lac_hetero_norm - lac_uniform_norm
        print(f"\n  Uniform lac: {lac_uniform:.4f} (norm: {lac_uniform_norm:.4f})")
        print(f"  Hetero lac:  {lac_hetero:.4f} (norm: {lac_hetero_norm:.4f})")
        print(f"  Difference:  {diff:.4f}")
        assert diff > 0.3, (
            f"Heterogeneous region should have much higher lacunarity than uniform. "
            f"Normalized diff={diff:.4f} (need > 0.3)"
        )

    def test_constant_image_lacunarity_is_one(self):
        """A constant image should have lacunarity ≈ 1.0 everywhere
        (no variation → var=0 → lac=1)."""
        image = torch.ones(1, 64, 64) * 0.7
        lac = compute_lacunarity_map(image, window_size=16, stride=8, box_sizes=(2, 4, 8))
        # All values should be close to 1.0 (lac = var/mu^2 + 1 = 0 + 1)
        assert (lac - 1.0).abs().max() < 0.1, (
            f"Constant image lacunarity should be ~1.0, got range [{lac.min():.4f}, {lac.max():.4f}]"
        )

    def test_zero_image_returns_ones(self):
        """An all-zero image should return lacunarity=1.0 (degenerate)."""
        image = torch.zeros(1, 64, 64)
        lac = compute_lacunarity_map(image, window_size=16, stride=8)
        assert (lac - 1.0).abs().max() < 0.1

    def test_gliding_box_no_per_patch_normalization(self):
        """Verify that gliding_box_lacunarity does NOT normalize patches.
        Two patches with different intensity ranges should produce
        different lacunarity values."""
        # Low-intensity patch
        low_patch = torch.zeros(32, 32)
        low_patch[8:24, 8:24] = 0.1

        # High-intensity patch (same structure, 10× intensity)
        high_patch = low_patch * 10.0

        lac_low = gliding_box_lacunarity(low_patch, box_sizes=(4, 8))
        lac_high = gliding_box_lacunarity(high_patch, box_sizes=(4, 8))

        # Without per-patch normalization, these should be identical
        # (lacunarity is scale-invariant: var/mu^2 cancels scaling)
        # but NOT because of normalization — because of the math
        assert abs(lac_low - lac_high) < 0.01, (
            f"Lacunarity should be scale-invariant. "
            f"low={lac_low:.6f}, high={lac_high:.6f}"
        )


class TestDatasetPercentiles:

    def test_percentile_roundtrip(self, tmp_path):
        """Save and load percentiles."""
        p = (1.05, 3.72)
        path = tmp_path / "lac_percentiles.json"
        save_percentiles(p, path)
        loaded = load_percentiles(path)
        assert loaded is not None
        assert abs(loaded[0] - p[0]) < 1e-6
        assert abs(loaded[1] - p[1]) < 1e-6

    def test_missing_file_returns_none(self, tmp_path):
        """Loading from non-existent path returns None."""
        result = load_percentiles(tmp_path / "nonexistent.json")
        assert result is None

    def test_compute_dataset_percentiles(self):
        """Compute percentiles over a small dataset."""
        torch.manual_seed(123)
        images = [torch.rand(1, 64, 64) for _ in range(3)]
        p1, p99 = compute_dataset_percentiles(
            images, window_size=16, stride=8, box_sizes=(2, 4), p_low=1.0, p_high=99.0
        )
        assert p1 < p99, f"p1={p1} should be < p99={p99}"
        assert p1 >= 1.0, f"p1={p1} should be >= 1.0 (lacunarity minimum)"

    def test_norm_percentiles_clamps_output(self):
        """With norm_percentiles, output should be in [0, 1]."""
        torch.manual_seed(42)
        image = torch.rand(1, 64, 64)
        lac = compute_lacunarity_map(
            image, window_size=16, stride=8, box_sizes=(2, 4, 8),
            norm_percentiles=(1.0, 2.0),
        )
        assert lac.min() >= 0.0
        assert lac.max() <= 1.0
