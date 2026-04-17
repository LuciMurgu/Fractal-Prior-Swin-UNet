"""Unit tests for percolation connectivity prior."""

import torch
import pytest

from fractal_swin_unet.fractal.percolation import (
    compute_percolation_map,
    percolation_critical_threshold,
    _connected_components_largest_ratio,
)


class TestConnectedComponentsRatio:
    def test_empty_image(self):
        binary = torch.zeros(16, 16)
        assert _connected_components_largest_ratio(binary) == 0.0

    def test_fully_connected(self):
        binary = torch.ones(16, 16)
        assert _connected_components_largest_ratio(binary) == pytest.approx(1.0)

    def test_two_components(self):
        binary = torch.zeros(16, 16)
        binary[0:4, 0:4] = 1  # 16 pixels
        binary[10:14, 10:14] = 1  # 16 pixels
        ratio = _connected_components_largest_ratio(binary)
        assert ratio == pytest.approx(0.5, abs=0.01)  # 50% each

    def test_one_large_one_small(self):
        binary = torch.zeros(16, 16)
        binary[0:8, 0:8] = 1  # 64 pixels
        binary[14, 14] = 1  # 1 pixel
        ratio = _connected_components_largest_ratio(binary)
        assert ratio > 0.9  # dominant component


class TestPercolationCriticalThreshold:
    def test_uniform_intensity(self):
        """Uniform image has no structure → high threshold."""
        patch = torch.ones(32, 32) * 0.5
        p_c = percolation_critical_threshold(patch)
        assert p_c == 1.0  # normalized uniform → max threshold

    def test_dense_structure(self):
        """High-contrast image with broad structure → high critical threshold.

        Dense regions percolate even at strict (high) thresholds.
        """
        patch = torch.zeros(32, 32)
        patch[:, :24] = 1.0  # 75% of pixels are bright
        p_c = percolation_critical_threshold(patch)
        # The bright region forms a spanning cluster at high threshold
        assert p_c >= 0.5

    def test_sparse_structure(self):
        """Only a few pixels are bright → high threshold needed."""
        patch = torch.zeros(32, 32)
        patch[15, 15] = 1.0  # single bright pixel
        p_c = percolation_critical_threshold(patch)
        # After normalization, single pixel can't span much
        assert p_c > 0.0

    def test_value_range(self):
        torch.manual_seed(42)
        patch = torch.rand(32, 32)
        p_c = percolation_critical_threshold(patch)
        assert 0.0 <= p_c <= 1.0


class TestPercolationMap:
    def test_shape(self):
        image = torch.rand(3, 64, 64)
        perc_map = compute_percolation_map(image, window_size=16)
        assert perc_map.shape == (64, 64)

    def test_value_range(self):
        image = torch.rand(3, 64, 64)
        perc_map = compute_percolation_map(image, window_size=16)
        assert perc_map.min() >= 0.0
        assert perc_map.max() <= 1.0

    def test_grayscale_input(self):
        image = torch.rand(64, 64)
        perc_map = compute_percolation_map(image, window_size=16)
        assert perc_map.shape == (64, 64)

    def test_single_channel(self):
        image = torch.rand(1, 32, 32)
        perc_map = compute_percolation_map(image, window_size=16)
        assert perc_map.shape == (32, 32)

    def test_small_window(self):
        """Should work with window_size smaller than image."""
        image = torch.rand(3, 64, 64)
        perc_map = compute_percolation_map(image, window_size=8)
        assert perc_map.shape == (64, 64)

    def test_dtype(self):
        image = torch.rand(3, 32, 32)
        perc_map = compute_percolation_map(image, window_size=16)
        assert perc_map.dtype == torch.float32
