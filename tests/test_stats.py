"""Tests for paired statistical helpers."""

import numpy as np
import pytest

from fractal_swin_unet.stats import (
    bootstrap_ci,
    cohens_d,
    format_stats_row,
    paired_wilcoxon,
)


class TestPairedWilcoxon:

    def test_identical_arrays(self):
        """Identical arrays → p=1.0, ns."""
        a = [0.8, 0.85, 0.82, 0.88, 0.81]
        b = [0.8, 0.85, 0.82, 0.88, 0.81]
        result = paired_wilcoxon(a, b)
        assert result["p_value"] == 1.0
        assert result["significant"] == "ns"
        assert result["delta_mean"] == 0.0

    def test_clearly_different(self):
        """Clearly different arrays → p < 0.05."""
        rng = np.random.default_rng(42)
        a = rng.normal(0.85, 0.02, size=50)
        b = rng.normal(0.70, 0.02, size=50)
        result = paired_wilcoxon(a, b, alternative="greater")
        assert result["p_value"] < 0.001
        assert result["significant"] == "***"
        assert result["delta_mean"] > 0.1

    def test_returns_expected_keys(self):
        a = [0.8, 0.9, 0.85]
        b = [0.7, 0.75, 0.72]
        result = paired_wilcoxon(a, b)
        expected_keys = {"statistic", "p_value", "significant", "n",
                         "delta_mean", "delta_std", "a_mean", "a_std",
                         "b_mean", "b_std"}
        assert set(result.keys()) == expected_keys

    def test_small_n(self):
        """n < 3 → p=1.0 (can't run test)."""
        result = paired_wilcoxon([0.8, 0.9], [0.7, 0.75])
        assert result["p_value"] == 1.0
        assert result["n"] == 2


class TestCohensD:

    def test_zero_difference(self):
        a = [0.8, 0.8, 0.8]
        b = [0.8, 0.8, 0.8]
        assert cohens_d(a, b) == 0.0

    def test_large_effect(self):
        """Large consistent difference → d > 0.8."""
        a = [0.90, 0.93, 0.91, 0.94, 0.89]
        b = [0.70, 0.71, 0.72, 0.73, 0.69]
        d = cohens_d(a, b)
        assert d > 0.8, f"Expected large effect, got d={d}"

    def test_sign(self):
        """d > 0 when a > b, d < 0 when a < b."""
        a = [0.9, 0.85, 0.88]
        b = [0.7, 0.75, 0.72]
        assert cohens_d(a, b) > 0
        assert cohens_d(b, a) < 0

    def test_moderate_effect(self):
        """Moderate effect size with some overlap."""
        rng = np.random.default_rng(42)
        a = rng.normal(0.80, 0.05, size=30)
        b = rng.normal(0.77, 0.05, size=30)
        d = cohens_d(a, b)
        assert 0.2 < abs(d) < 2.0


class TestBootstrapCI:

    def test_zero_difference(self):
        """Identical arrays → CI includes 0."""
        a = [0.8, 0.85, 0.82, 0.88, 0.81]
        b = [0.8, 0.85, 0.82, 0.88, 0.81]
        lo, hi = bootstrap_ci(a, b)
        assert lo == 0.0
        assert hi == 0.0

    def test_positive_difference(self):
        """Clear positive difference → CI entirely above 0."""
        rng = np.random.default_rng(42)
        a = rng.normal(0.85, 0.02, size=50)
        b = rng.normal(0.75, 0.02, size=50)
        lo, hi = bootstrap_ci(a, b, n_bootstrap=5000)
        assert lo > 0, f"CI lower bound {lo} should be > 0"
        assert hi > lo

    def test_contains_observed_mean(self):
        """The observed mean difference should be within the CI."""
        a = np.array([0.8, 0.85, 0.9, 0.82, 0.88])
        b = np.array([0.7, 0.75, 0.8, 0.72, 0.78])
        observed = (a - b).mean()
        lo, hi = bootstrap_ci(a, b, n_bootstrap=10000)
        assert lo <= observed <= hi

    def test_reproducible(self):
        """Same seed → same CI."""
        a = [0.8, 0.85, 0.9]
        b = [0.7, 0.75, 0.8]
        ci1 = bootstrap_ci(a, b, seed=123)
        ci2 = bootstrap_ci(a, b, seed=123)
        assert ci1 == ci2


class TestFormatStatsRow:

    def test_returns_all_fields(self):
        a = [0.8, 0.85, 0.9, 0.82, 0.88]
        b = [0.7, 0.75, 0.8, 0.72, 0.78]
        row = format_stats_row("Test vs Base", a, b, metric_name="Dice")
        expected_keys = {"label", "metric", "n", "a_mean", "a_std",
                         "b_mean", "b_std", "delta_mean", "delta_std",
                         "p_value", "significant", "cohens_d", "ci_lo", "ci_hi"}
        assert set(row.keys()) == expected_keys
        assert row["label"] == "Test vs Base"
        assert row["metric"] == "Dice"
        assert row["n"] == 5
