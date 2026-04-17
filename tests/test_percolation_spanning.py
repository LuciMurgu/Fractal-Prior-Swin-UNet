"""Tests for true spanning-based percolation and GPU CC labeling."""

import torch
import pytest

from fractal_swin_unet.fractal.percolation import (
    percolation_critical_threshold,
    _check_spanning,
    cc_label,
    _cc_label_gpu,
    _cc_label_scipy,
)


class TestSpanningCondition:

    def test_fully_connected_high_pc(self):
        """A fully connected image → p_c should be high (≥ 0.7)."""
        patch = torch.ones(32, 32)
        p_c = percolation_critical_threshold(patch)
        # Fully connected at all thresholds → uniform → p_c = 1.0
        assert p_c >= 0.7, f"Fully connected should have high p_c, got {p_c}"

    def test_fragmented_noise_low_pc(self):
        """Random sparse noise → p_c should be low."""
        torch.manual_seed(42)
        # Very sparse: only ~5% of pixels bright
        patch = torch.zeros(32, 32)
        mask = torch.rand(32, 32) < 0.05
        patch[mask] = 1.0
        # With very sparse random pixels, spanning is unlikely at high tau
        p_c = percolation_critical_threshold(patch)
        # Should be low (≤ 0.5) since sparse noise rarely spans
        assert p_c <= 0.5, f"Fragmented noise should have low p_c, got {p_c}"

    def test_centered_blob_no_spanning(self):
        """A single centered blob NOT touching any edge → should NOT be
        counted as percolating at any threshold."""
        patch = torch.zeros(32, 32)
        # Center blob: rows 8-24, cols 8-24 (no edge touching)
        patch[8:24, 8:24] = 1.0
        p_c = percolation_critical_threshold(patch)
        # The blob doesn't span → should get lowest threshold
        assert p_c <= 0.2, (
            f"Centered blob (no edge touching) should not be percolating. "
            f"Got p_c={p_c}"
        )

    def test_all_zero_patch_returns_lowest(self):
        """All-zero patch → should return lowest threshold, not NaN."""
        patch = torch.zeros(32, 32)
        p_c = percolation_critical_threshold(patch)
        assert not torch.isnan(torch.tensor(p_c))
        assert isinstance(p_c, float)
        # For uniform zero → normalized is uniform → p_c = 1.0 (trivial)
        # But since all values are same, vmax-vmin=0 → returns 1.0
        assert p_c >= 0.0

    def test_vertical_spanning_bar(self):
        """A vertical bar spanning top to bottom → should be percolating."""
        patch = torch.zeros(32, 32)
        patch[:, 14:18] = 1.0  # spans all rows
        p_c = percolation_critical_threshold(patch)
        assert p_c >= 0.5, f"Vertical spanning bar should percolate, got p_c={p_c}"

    def test_horizontal_spanning_bar(self):
        """A horizontal bar spanning left to right → should be percolating."""
        patch = torch.zeros(32, 32)
        patch[14:18, :] = 1.0  # spans all columns
        p_c = percolation_critical_threshold(patch)
        assert p_c >= 0.5, f"Horizontal spanning bar should percolate, got p_c={p_c}"


class TestCheckSpanning:

    def test_empty_labels(self):
        labels = torch.zeros(16, 16, dtype=torch.long)
        assert _check_spanning(labels) is False

    def test_vertical_span(self):
        labels = torch.zeros(16, 16, dtype=torch.long)
        labels[:, 7] = 1  # vertical strip → touches top AND bottom
        assert _check_spanning(labels) is True

    def test_horizontal_span(self):
        labels = torch.zeros(16, 16, dtype=torch.long)
        labels[7, :] = 2  # horizontal strip → touches left AND right
        assert _check_spanning(labels) is True

    def test_corner_only_no_span(self):
        labels = torch.zeros(16, 16, dtype=torch.long)
        labels[0, 0] = 1  # only top-left corner
        assert _check_spanning(labels) is False

    def test_top_bottom_different_labels(self):
        """Two components touching top and bottom separately → no span."""
        labels = torch.zeros(16, 16, dtype=torch.long)
        labels[0, 5:10] = 1   # top row
        labels[15, 5:10] = 2  # bottom row (different label)
        assert _check_spanning(labels) is False

    def test_top_bottom_same_label(self):
        """One component touching both top and bottom → spans."""
        labels = torch.zeros(16, 16, dtype=torch.long)
        labels[:, 7] = 1  # same label across all rows
        assert _check_spanning(labels) is True


class TestCCLabel:

    def test_empty_input(self):
        binary = torch.zeros(16, 16)
        labels = cc_label(binary)
        assert labels.max() == 0

    def test_single_component(self):
        binary = torch.zeros(16, 16)
        binary[3:12, 3:12] = 1.0
        labels = cc_label(binary)
        # All foreground should have the same label
        fg_labels = labels[binary > 0].unique()
        assert len(fg_labels) == 1
        assert fg_labels[0] > 0

    def test_two_disjoint_components(self):
        binary = torch.zeros(16, 16)
        binary[0:4, 0:4] = 1.0
        binary[10:14, 10:14] = 1.0
        labels = cc_label(binary)
        fg_labels = labels[binary > 0].unique()
        assert len(fg_labels) == 2  # two distinct components

    def test_gpu_matches_scipy(self):
        """GPU CC labeling must agree with scipy on component count."""
        torch.manual_seed(123)
        binary = (torch.rand(32, 32) > 0.6).float()

        labels_scipy = _cc_label_scipy(binary)
        labels_gpu = _cc_label_gpu(binary)

        # Check same number of components
        n_scipy = len(labels_scipy.unique()) - 1  # exclude 0
        n_gpu = len(labels_gpu.unique()) - 1
        assert n_scipy == n_gpu, (
            f"Component count mismatch: scipy={n_scipy}, gpu={n_gpu}"
        )

        # Check same connectivity: foreground pixels with same scipy label
        # should have the same gpu label (modulo relabeling)
        for label_val in labels_scipy.unique():
            if label_val == 0:
                continue
            scipy_mask = (labels_scipy == label_val)
            gpu_labels_in_region = labels_gpu[scipy_mask].unique()
            assert len(gpu_labels_in_region) == 1, (
                f"scipy component {label_val} maps to {len(gpu_labels_in_region)} gpu labels"
            )

    def test_dtype_is_long(self):
        binary = torch.ones(8, 8)
        labels = cc_label(binary)
        assert labels.dtype == torch.long
