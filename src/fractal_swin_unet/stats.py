"""Paired statistical test utilities for journal-grade evidence.

Provides reusable helpers for per-image paired comparisons:
- Wilcoxon signed-rank test
- Cohen's d effect size
- Bootstrap confidence intervals

All functions accept numpy arrays or lists. When scipy is unavailable,
a pure-numpy fallback is used for the Wilcoxon test (warns).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def paired_wilcoxon(
    a: np.ndarray | list,
    b: np.ndarray | list,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """Paired Wilcoxon signed-rank test.

    Args:
        a: Per-image metrics for condition A (e.g. fractal).
        b: Per-image metrics for condition B (e.g. baseline).
        alternative: 'two-sided', 'greater', or 'less'.

    Returns:
        dict with keys: statistic, p_value, significant, n, delta_mean,
        delta_std, a_mean, a_std, b_mean, b_std.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert len(a) == len(b), f"Length mismatch: {len(a)} vs {len(b)}"
    n = len(a)
    diff = a - b

    if n < 3 or np.all(diff == 0):
        statistic, p_value = 0.0, 1.0
    else:
        try:
            from scipy.stats import wilcoxon as _wilcoxon
            statistic, p_value = _wilcoxon(a, b, alternative=alternative)
        except ImportError:
            import warnings
            warnings.warn("scipy not available; p-value set to NaN")
            statistic, p_value = 0.0, float("nan")

    sig = _significance_label(p_value)

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": sig,
        "n": n,
        "delta_mean": float(diff.mean()),
        "delta_std": float(diff.std(ddof=1)) if n > 1 else 0.0,
        "a_mean": float(a.mean()),
        "a_std": float(a.std(ddof=1)) if n > 1 else 0.0,
        "b_mean": float(b.mean()),
        "b_std": float(b.std(ddof=1)) if n > 1 else 0.0,
    }


def cohens_d(a: np.ndarray | list, b: np.ndarray | list) -> float:
    """Paired Cohen's d effect size.

    d = mean(a - b) / std(a - b, ddof=1)

    Interpretation (Cohen 1988):
        |d| < 0.2  → negligible
        |d| < 0.5  → small
        |d| < 0.8  → medium
        |d| >= 0.8 → large

    Args:
        a: Per-image metrics for condition A.
        b: Per-image metrics for condition B.

    Returns:
        Cohen's d (positive = A > B).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
    std = diff.std(ddof=1)
    if std < 1e-15:
        return 0.0
    return float(diff.mean() / std)


def bootstrap_ci(
    a: np.ndarray | list,
    b: np.ndarray | list,
    n_bootstrap: int = 5000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval on the mean difference (a - b).

    Args:
        a: Per-image metrics for condition A.
        b: Per-image metrics for condition B.
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (default 0.95).
        seed: Random seed for reproducibility.

    Returns:
        (lower, upper) confidence interval bounds.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
    n = len(diff)

    if n < 2:
        return (float(diff.mean()), float(diff.mean()))

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = diff[idx].mean()

    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_means, 100 * alpha))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha)))
    return (lo, hi)


def format_stats_row(
    label: str,
    a: np.ndarray | list,
    b: np.ndarray | list,
    metric_name: str = "Dice",
    n_bootstrap: int = 5000,
) -> dict[str, Any]:
    """Compute all statistics for a single comparison row.

    Args:
        label: Row label (e.g. "OPT-D vs Baseline").
        a: Per-image metrics for the experimental condition.
        b: Per-image metrics for the baseline condition.
        metric_name: Name of the metric (for labeling).
        n_bootstrap: Number of bootstrap resamples.

    Returns:
        dict with all statistical fields for the row.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    wil = paired_wilcoxon(a, b, alternative="greater")
    d = cohens_d(a, b)
    ci_lo, ci_hi = bootstrap_ci(a, b, n_bootstrap=n_bootstrap)

    return {
        "label": label,
        "metric": metric_name,
        "n": wil["n"],
        "a_mean": wil["a_mean"],
        "a_std": wil["a_std"],
        "b_mean": wil["b_mean"],
        "b_std": wil["b_std"],
        "delta_mean": wil["delta_mean"],
        "delta_std": wil["delta_std"],
        "p_value": wil["p_value"],
        "significant": wil["significant"],
        "cohens_d": d,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
    }


def _significance_label(p: float) -> str:
    """Convert p-value to significance label."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"
