"""Unit tests for journal-grade metrics (Se, Sp, Acc, F1, AUC-ROC)."""

import torch
import pytest

from fractal_swin_unet.metrics import (
    auroc,
    cldice_score,
    dice_score,
    f1_score,
    pixel_accuracy,
    sensitivity,
    specificity,
)


def _make_perfect_pair():
    """Create perfect prediction (pred == target)."""
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 2:6, 2:6] = 1.0  # 16 positive pixels
    return target.clone(), target


def _make_all_wrong_pair():
    """Create inverted prediction (pred = 1 - target)."""
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 2:6, 2:6] = 1.0
    pred = 1.0 - target
    return pred, target


def _make_known_pair():
    """Create a case with known TP=4, FP=2, FN=1, TN=9."""
    # 4x4 grid = 16 pixels
    pred = torch.zeros(1, 1, 4, 4)
    target = torch.zeros(1, 1, 4, 4)

    # Ground truth positive: (0,0), (0,1), (1,0), (1,1), (2,0)  → 5 pos
    target[0, 0, 0, 0] = 1
    target[0, 0, 0, 1] = 1
    target[0, 0, 1, 0] = 1
    target[0, 0, 1, 1] = 1
    target[0, 0, 2, 0] = 1

    # Predicted positive: (0,0), (0,1), (1,0), (1,1), (3,0), (3,1)  → 6 pos
    pred[0, 0, 0, 0] = 1
    pred[0, 0, 0, 1] = 1
    pred[0, 0, 1, 0] = 1
    pred[0, 0, 1, 1] = 1
    pred[0, 0, 3, 0] = 1
    pred[0, 0, 3, 1] = 1

    # TP=4, FP=2, FN=1, TN=9
    return pred, target


class TestSensitivity:
    def test_perfect_predictions(self):
        pred, target = _make_perfect_pair()
        se = sensitivity(pred, target)
        assert se.item() == pytest.approx(1.0, abs=1e-4)

    def test_all_wrong(self):
        pred, target = _make_all_wrong_pair()
        se = sensitivity(pred, target)
        assert se.item() < 0.01  # near zero (only eps)

    def test_known_case(self):
        pred, target = _make_known_pair()
        # TP=4, FN=1 → Se = 4/5 = 0.8
        se = sensitivity(pred, target)
        assert se.item() == pytest.approx(0.8, abs=0.02)


class TestSpecificity:
    def test_perfect_predictions(self):
        pred, target = _make_perfect_pair()
        sp = specificity(pred, target)
        assert sp.item() == pytest.approx(1.0, abs=1e-4)

    def test_known_case(self):
        pred, target = _make_known_pair()
        # TN=9, FP=2 → Sp = 9/11 = 0.818
        sp = specificity(pred, target)
        assert sp.item() == pytest.approx(9.0 / 11.0, abs=0.02)


class TestAccuracy:
    def test_perfect_predictions(self):
        pred, target = _make_perfect_pair()
        acc = pixel_accuracy(pred, target)
        assert acc.item() == pytest.approx(1.0, abs=1e-4)

    def test_known_case(self):
        pred, target = _make_known_pair()
        # Correct = TP + TN = 4 + 9 = 13 out of 16
        acc = pixel_accuracy(pred, target)
        assert acc.item() == pytest.approx(13.0 / 16.0, abs=0.02)


class TestF1:
    def test_perfect_predictions(self):
        pred, target = _make_perfect_pair()
        score = f1_score(pred, target)
        assert score.item() == pytest.approx(1.0, abs=1e-4)

    def test_known_case(self):
        pred, target = _make_known_pair()
        # Precision = 4/6 = 0.667, Recall = 4/5 = 0.8
        # F1 = 2 * 0.667 * 0.8 / (0.667 + 0.8) = 0.727
        score = f1_score(pred, target)
        assert score.item() == pytest.approx(0.727, abs=0.03)


class TestAUROC:
    def test_perfect_separation(self):
        """Perfect classifier: positive pixels have prob=1, negative have prob=0."""
        target = torch.zeros(1, 1, 8, 8)
        target[0, 0, 2:6, 2:6] = 1.0
        probs = target.clone()  # perfect
        auc = auroc(probs, target)
        assert auc == pytest.approx(1.0, abs=0.02)

    def test_random_classifier(self):
        """Random classifier should give AUC ≈ 0.5."""
        torch.manual_seed(42)
        target = (torch.rand(1, 1, 32, 32) > 0.5).float()
        probs = torch.rand(1, 1, 32, 32)
        auc = auroc(probs, target)
        assert 0.3 < auc < 0.7  # within random range

    def test_with_mask(self):
        """AUC computed only within FOV mask."""
        target = torch.zeros(1, 1, 8, 8)
        target[0, 0, 2:6, 2:6] = 1.0
        probs = target.clone()
        mask = torch.ones(1, 1, 8, 8)
        mask[0, 0, :2, :] = 0  # mask out top 2 rows
        auc = auroc(probs, target, mask=mask)
        assert auc == pytest.approx(1.0, abs=0.02)


class TestWithFOVMask:
    def test_sensitivity_with_mask(self):
        pred, target = _make_known_pair()
        mask = torch.ones_like(pred)
        se_no_mask = sensitivity(pred, target)
        se_with_mask = sensitivity(pred, target, mask=mask)
        assert se_no_mask.item() == pytest.approx(se_with_mask.item(), abs=1e-4)

    def test_metrics_consistent_dtype(self):
        """All metrics should work with float16 inputs."""
        pred = torch.ones(1, 1, 4, 4, dtype=torch.float16)
        target = torch.ones(1, 1, 4, 4, dtype=torch.float16)
        # Should not raise
        sensitivity(pred, target)
        specificity(pred, target)
        pixel_accuracy(pred, target)
        f1_score(pred, target)
