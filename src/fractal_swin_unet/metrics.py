"""Metrics for segmentation."""

from __future__ import annotations

import torch

from .losses import soft_skeletonize


def dice_score(
    probs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute Dice score from probabilities.

    Args:
        probs: Probability tensor of shape (B, 1, H, W).
        targets: Binary targets of shape (B, 1, H, W).
        eps: Small value for numerical stability.

    Returns:
        Scalar Dice score tensor.
    """

    if mask is not None:
        mask = mask.float()
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        probs = probs * mask
        targets = targets * mask

    intersection = (probs * targets).sum(dim=(2, 3))
    denom = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2 * intersection + eps) / (denom + eps)
    return dice.mean()


def cldice_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    iters: int = 10,
    eps: float = 1e-6,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute clDice (connectivity-preserving Dice) on binary masks.

    Unlike the soft clDice *loss*, this operates on hard binary predictions
    and returns a *score* (higher is better).

    Degenerate-case handling: if either skeleton is empty (sum ≤ eps),
    the corresponding tprec/tsens is set to 0, yielding clDice = 0 for
    that sample. This prevents empty predictions from scoring 1.0.

    Args:
        preds: Binary predictions (B, 1, H, W), values in {0, 1}.
        targets: Binary ground truth (B, 1, H, W).
        iters: Skeletonization iterations.
        eps: Numerical stability.
        mask: Optional FOV mask (B, 1, H, W).

    Returns:
        Scalar clDice score tensor.
    """
    preds = preds.float()
    targets = targets.float()

    if mask is not None:
        mask = mask.float()
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        preds = preds * mask
        targets = targets * mask

    sp = soft_skeletonize(preds, iters)
    sg = soft_skeletonize(targets, iters)

    sp_sum = sp.sum(dim=(2, 3))  # (B, 1)
    sg_sum = sg.sum(dim=(2, 3))  # (B, 1)

    # Topology precision: skeleton(pred) covered by GT
    # If pred skeleton is empty → tprec = 0 (no topology to evaluate)
    tprec = torch.where(
        sp_sum > eps,
        (sp * targets).sum(dim=(2, 3)) / (sp_sum + eps),
        torch.zeros_like(sp_sum),
    )
    # Topology sensitivity: skeleton(GT) covered by pred
    # If GT skeleton is empty → tsens = 0
    tsens = torch.where(
        sg_sum > eps,
        (sg * preds).sum(dim=(2, 3)) / (sg_sum + eps),
        torch.zeros_like(sg_sum),
    )

    # Harmonic mean; degenerate (both 0) → clDice = 0
    denom = tprec + tsens
    cldice = torch.where(
        denom > eps,
        2 * tprec * tsens / (denom + eps),
        torch.zeros_like(denom),
    )
    return cldice.mean()


def sensitivity(
    preds: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sensitivity (recall / true positive rate): TP / (TP + FN).

    Args:
        preds: Binary predictions (B, 1, H, W).
        targets: Binary ground truth (B, 1, H, W).
        eps: Numerical stability.
        mask: Optional FOV mask.

    Returns:
        Scalar sensitivity tensor.
    """
    preds, targets = preds.float(), targets.float()
    if mask is not None:
        mask = mask.float()
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        preds = preds * mask
        targets = targets * mask

    tp = (preds * targets).sum()
    fn = ((1 - preds) * targets).sum()
    return (tp + eps) / (tp + fn + eps)


def specificity(
    preds: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Specificity (true negative rate): TN / (TN + FP).

    Args:
        preds: Binary predictions (B, 1, H, W).
        targets: Binary ground truth (B, 1, H, W).
        eps: Numerical stability.
        mask: Optional FOV mask.

    Returns:
        Scalar specificity tensor.
    """
    preds, targets = preds.float(), targets.float()
    if mask is not None:
        mask = mask.float()
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        preds = preds * mask
        targets = targets * mask

    tn = ((1 - preds) * (1 - targets)).sum()
    fp = (preds * (1 - targets)).sum()
    # When using FOV mask, exclude masked-out pixels from TN count
    if mask is not None:
        tn = tn - ((1 - mask).sum())  # masked pixels are neither TN nor FP
        tn = tn.clamp(min=0)
    return (tn + eps) / (tn + fp + eps)


def pixel_accuracy(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pixel-level accuracy: (TP + TN) / total.

    Args:
        preds: Binary predictions (B, 1, H, W).
        targets: Binary ground truth (B, 1, H, W).
        mask: Optional FOV mask.

    Returns:
        Scalar accuracy tensor.
    """
    preds, targets = preds.float(), targets.float()
    if mask is not None:
        mask = mask.float()
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        preds = preds * mask
        targets = targets * mask
        total = mask.sum()
    else:
        total = torch.tensor(preds.numel(), device=preds.device, dtype=preds.dtype)

    correct = ((preds == targets).float() * (mask if mask is not None else 1.0)).sum()
    return correct / total.clamp(min=1)


def f1_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """F1 score: harmonic mean of precision and recall.

    Args:
        preds: Binary predictions (B, 1, H, W).
        targets: Binary ground truth (B, 1, H, W).
        eps: Numerical stability.
        mask: Optional FOV mask.

    Returns:
        Scalar F1 tensor.
    """
    preds, targets = preds.float(), targets.float()
    if mask is not None:
        mask = mask.float()
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        preds = preds * mask
        targets = targets * mask

    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return (2 * precision * recall + eps) / (precision + recall + eps)


def auroc(
    probs: torch.Tensor,
    targets: torch.Tensor,
    num_thresholds: int = 200,
    mask: torch.Tensor | None = None,
) -> float:
    """Area under the ROC curve via trapezoidal integration.

    Args:
        probs: Probability tensor (B, 1, H, W) in [0, 1].
        targets: Binary ground truth (B, 1, H, W).
        num_thresholds: Number of threshold points for the ROC curve.
        mask: Optional FOV mask.

    Returns:
        AUC-ROC as a float.
    """
    probs, targets = probs.float(), targets.float()
    if mask is not None:
        mask = mask.float()
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        # Flatten only within FOV
        valid = mask.flatten().bool()
        p_flat = probs.flatten()[valid]
        t_flat = targets.flatten()[valid]
    else:
        p_flat = probs.flatten()
        t_flat = targets.flatten()

    # Sort by decreasing probability for efficient TPR/FPR computation
    thresholds = torch.linspace(1.0, 0.0, num_thresholds, device=p_flat.device)
    total_pos = t_flat.sum().clamp(min=1)
    total_neg = (1 - t_flat).sum().clamp(min=1)

    tprs = []
    fprs = []
    for tau in thresholds:
        pred = (p_flat >= tau).float()
        tp = (pred * t_flat).sum()
        fp = (pred * (1 - t_flat)).sum()
        tprs.append((tp / total_pos).item())
        fprs.append((fp / total_neg).item())

    # Trapezoidal integration
    auc = 0.0
    for i in range(1, len(fprs)):
        auc += 0.5 * (tprs[i] + tprs[i - 1]) * (fprs[i] - fprs[i - 1])
    return abs(auc)
