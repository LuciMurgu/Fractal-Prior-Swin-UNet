"""Regression tests for clDice metric/loss degenerate-case fix.

The original implementation used symmetric epsilon in the harmonic mean,
which caused clDice = 1.0 when predictions were empty (skeleton sum ≈ 0).
These tests ensure the fix is correct and doesn't regress.
"""

import torch
import pytest

from fractal_swin_unet.metrics import cldice_score
from fractal_swin_unet.losses import cldice_loss


def _vessel_target():
    """Create a non-empty vessel target mask."""
    target = torch.zeros(1, 1, 32, 32)
    # Horizontal vessel
    target[0, 0, 14:18, 4:28] = 1.0
    # Vertical branch
    target[0, 0, 8:24, 14:18] = 1.0
    return target


class TestClDiceMetricFix:
    """Regression tests for cldice_score degenerate-case handling."""

    def test_empty_prediction_non_empty_target_score_zero(self):
        """Empty prediction + non-empty target → score MUST be 0.0.

        This was the critical bug: old code returned 1.0 here.
        """
        preds = torch.zeros(1, 1, 32, 32)
        targets = _vessel_target()
        score = cldice_score(preds, targets)
        assert score.item() == pytest.approx(0.0, abs=1e-4), \
            f"Empty prediction should score 0.0, got {score.item()}"

    def test_perfect_prediction_score_near_one(self):
        """Perfect prediction → score should be ~1.0."""
        targets = _vessel_target()
        preds = targets.clone()
        score = cldice_score(preds, targets)
        assert score.item() == pytest.approx(1.0, abs=0.05), \
            f"Perfect prediction should score ~1.0, got {score.item()}"

    def test_partial_overlap_between_zero_and_one(self):
        """Partial overlap → score should be between 0 and 1."""
        targets = _vessel_target()
        # Shift prediction by 4 pixels → partial overlap
        preds = torch.zeros_like(targets)
        preds[0, 0, 14:18, 8:32] = 1.0  # shifted horizontal
        score = cldice_score(preds, targets)
        assert 0.0 < score.item() < 1.0, \
            f"Partial overlap should be in (0, 1), got {score.item()}"

    def test_both_empty_score_zero(self):
        """Both empty → score should be 0.0 (nothing to evaluate)."""
        preds = torch.zeros(1, 1, 32, 32)
        targets = torch.zeros(1, 1, 32, 32)
        score = cldice_score(preds, targets)
        assert score.item() == pytest.approx(0.0, abs=1e-4)

    def test_empty_target_non_empty_pred_score_zero(self):
        """Non-empty pred + empty target → score should be 0.0."""
        preds = _vessel_target()
        targets = torch.zeros(1, 1, 32, 32)
        score = cldice_score(preds, targets)
        assert score.item() == pytest.approx(0.0, abs=1e-4)


class TestClDiceLossFix:
    """Regression tests for cldice_loss degenerate-case handling."""

    def test_empty_prediction_loss_one(self):
        """Empty prediction + non-empty target → loss MUST be 1.0.

        This was the critical bug: old code returned ~0.5 (rewarding collapse).
        """
        probs = torch.zeros(1, 1, 32, 32)
        targets = _vessel_target()
        loss = cldice_loss(probs, targets)
        assert loss.item() == pytest.approx(1.0, abs=0.05), \
            f"Empty prediction should have loss ~1.0, got {loss.item()}"

    def test_perfect_prediction_loss_near_zero(self):
        """Perfect prediction → loss should be near 0."""
        targets = _vessel_target()
        probs = targets.clone()
        loss = cldice_loss(probs, targets)
        assert loss.item() < 0.1, \
            f"Perfect prediction should have loss ~0.0, got {loss.item()}"

    def test_partial_overlap_loss_between_zero_and_one(self):
        """Partial overlap → loss should be between 0 and 1."""
        targets = _vessel_target()
        probs = torch.zeros_like(targets)
        probs[0, 0, 14:18, 8:32] = 1.0
        loss = cldice_loss(probs, targets)
        assert 0.0 < loss.item() < 1.0, \
            f"Partial overlap loss should be in (0, 1), got {loss.item()}"

    def test_loss_is_differentiable(self):
        """Loss must remain differentiable for gradient flow."""
        probs = torch.rand(1, 1, 32, 32, requires_grad=True)
        targets = _vessel_target()
        loss = cldice_loss(probs, targets)
        loss.backward()
        assert probs.grad is not None
        assert not torch.isnan(probs.grad).any()
