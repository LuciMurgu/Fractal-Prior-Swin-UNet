"""Threshold sweep utilities."""

from __future__ import annotations

from typing import Iterable, List, Dict, Any

import torch

from ..metrics import (
    auroc,
    cldice_score,
    dice_score,
    f1_score,
    pixel_accuracy,
    sensitivity,
    specificity,
)


def _ensure_grid(grid: Iterable[float]) -> List[float]:
    return [float(x) for x in grid]


def sweep_thresholds(
    probs: torch.Tensor,
    gts: torch.Tensor,
    grid: Iterable[float],
) -> Dict[str, Any]:
    """Sweep thresholds and compute Dice for each.

    Args:
        probs: Probability tensor of shape (B, 1, H, W).
        gts: Ground truth tensor of shape (B, 1, H, W).
        grid: Iterable of threshold values.

    Returns:
        Dict with grid, dice_per_tau, tau_star, best_dice, dice_at_0_5,
        plus clDice and journal metrics (Se, Sp, Acc, F1, AUC) at tau_star and 0.5.
    """

    grid_list = _ensure_grid(grid)
    dice_per_tau: List[float] = []

    for tau in grid_list:
        preds = (probs >= tau).float()
        dice_val = float(dice_score(preds, gts).item())
        dice_per_tau.append(dice_val)

    best_idx = max(range(len(dice_per_tau)), key=lambda i: (dice_per_tau[i], -grid_list[i]))
    tau_star = grid_list[best_idx]
    best_dice = dice_per_tau[best_idx]

    preds_at_tau_star = (probs >= tau_star).float()
    preds_at_0_5 = (probs >= 0.5).float()

    dice_at_0_5 = float(dice_score(preds_at_0_5, gts).item())
    cldice_at_tau_star = float(cldice_score(preds_at_tau_star, gts).item())
    cldice_at_0_5 = float(cldice_score(preds_at_0_5, gts).item())

    # Journal metrics at tau_star
    se_tau = float(sensitivity(preds_at_tau_star, gts).item())
    sp_tau = float(specificity(preds_at_tau_star, gts).item())
    acc_tau = float(pixel_accuracy(preds_at_tau_star, gts).item())
    f1_tau = float(f1_score(preds_at_tau_star, gts).item())

    # Journal metrics at 0.5
    se_05 = float(sensitivity(preds_at_0_5, gts).item())
    sp_05 = float(specificity(preds_at_0_5, gts).item())
    acc_05 = float(pixel_accuracy(preds_at_0_5, gts).item())
    f1_05 = float(f1_score(preds_at_0_5, gts).item())

    # AUC-ROC (threshold-independent)
    auc_val = auroc(probs, gts)

    return {
        "grid": grid_list,
        "dice_per_tau": dice_per_tau,
        "tau_star": tau_star,
        "best_dice": best_dice,
        "dice_at_0_5": dice_at_0_5,
        "best_cldice": cldice_at_tau_star,
        "cldice_at_0_5": cldice_at_0_5,
        "se_tau_star": se_tau,
        "sp_tau_star": sp_tau,
        "acc_tau_star": acc_tau,
        "f1_tau_star": f1_tau,
        "se_at_0_5": se_05,
        "sp_at_0_5": sp_05,
        "acc_at_0_5": acc_05,
        "f1_at_0_5": f1_05,
        "auc_roc": auc_val,
    }

