"""Test that eval loss computation is numerically stable.

Verifies that probs of exactly 0.0 and 1.0 (common after sigmoid with
large logits at late training epochs) don't produce NaN/Inf losses.
"""

import torch
import pytest

from fractal_swin_unet.eval import _safe_logit
from fractal_swin_unet.losses import dice_bce_loss


class TestSafeLogit:

    def test_exact_zero_probs(self):
        """probs=0.0 should produce finite logit, not -inf."""
        probs = torch.zeros(1, 1, 8, 8)
        logits = _safe_logit(probs)
        assert torch.isfinite(logits).all(), f"Got non-finite: {logits.min()}"

    def test_exact_one_probs(self):
        """probs=1.0 should produce finite logit, not +inf."""
        probs = torch.ones(1, 1, 8, 8)
        logits = _safe_logit(probs)
        assert torch.isfinite(logits).all(), f"Got non-finite: {logits.max()}"

    def test_mixed_extreme_probs(self):
        """Mix of 0.0 and 1.0 probs should all be finite."""
        probs = torch.tensor([[[[0.0, 1.0, 0.0, 1.0],
                                [1.0, 0.0, 0.5, 0.999]]]])
        logits = _safe_logit(probs)
        assert torch.isfinite(logits).all()

    def test_loss_finite_with_extreme_probs(self):
        """dice_bce_loss is finite when probs are 0.0 or 1.0."""
        # Simulate late-epoch model output: sigmoid(large) ≈ 1.0, sigmoid(-large) ≈ 0.0
        raw_logits = torch.tensor([[[[100.0, -100.0, 50.0, -50.0],
                                     [0.0, 100.0, -100.0, 0.5]]]])
        probs = torch.sigmoid(raw_logits)

        # probs will contain values exactly 0.0 and 1.0 in float32
        assert probs.min() == 0.0, "sigmoid(-100) should be exactly 0.0 in float32"
        assert probs.max() == 1.0, "sigmoid(100) should be exactly 1.0 in float32"

        targets = torch.tensor([[[[1.0, 0.0, 1.0, 0.0],
                                  [0.0, 1.0, 0.0, 1.0]]]])

        # Old way: probs.logit() → ±inf → NaN loss
        unsafe_logits = probs.logit()
        unsafe_loss = dice_bce_loss(unsafe_logits, targets)
        assert torch.isnan(unsafe_loss) or torch.isinf(unsafe_loss), \
            "Expected NaN/Inf from unsafe logit (this test validates the bug exists)"

        # New way: _safe_logit → finite
        safe_logits = _safe_logit(probs)
        safe_loss = dice_bce_loss(safe_logits, targets)
        assert torch.isfinite(safe_loss), f"Safe loss should be finite, got {safe_loss}"
        assert safe_loss.item() >= 0, f"Loss should be non-negative, got {safe_loss}"

    def test_normal_probs_unchanged(self):
        """Normal probabilities (away from 0/1) should be unchanged."""
        probs = torch.tensor([0.3, 0.5, 0.7])
        safe = _safe_logit(probs)
        direct = probs.logit()
        assert torch.allclose(safe, direct, atol=1e-4)
