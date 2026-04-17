"""Loss functions for segmentation."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F


def dice_loss(
    probs: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute soft Dice loss from probabilities.

    Args:
        probs: Probability tensor of shape (B, 1, H, W).
        targets: Binary targets of shape (B, 1, H, W).
        eps: Small value for numerical stability.

    Returns:
        Scalar Dice loss tensor.
    """

    intersection = (probs * targets).sum(dim=(2, 3))
    denom = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = 1 - (2 * intersection + smooth) / (denom + smooth + eps)
    return dice.mean()


def dice_bce_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute Dice + BCE loss.

    Args:
        logits: Logits tensor of shape (B, 1, H, W).
        targets: Binary targets of shape (B, 1, H, W).
        eps: Small value for numerical stability.

    Returns:
        Scalar loss tensor.
    """

    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    return bce + dice_loss(probs, targets, smooth=1.0, eps=eps)


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    targets = targets.float()
    pt = probs * targets + (1 - probs) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = -alpha_t * torch.pow(1 - pt, gamma) * torch.log(pt + eps)

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def _soft_erode(x: torch.Tensor) -> torch.Tensor:
    return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)


def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def _soft_open(x: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(x))


def soft_skeletonize(x: torch.Tensor, iters: int) -> torch.Tensor:
    skel = F.relu(x - _soft_open(x))
    for _ in range(max(iters - 1, 0)):
        x = _soft_erode(x)
        delta = F.relu(x - _soft_open(x))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def cldice_loss(
    probs: torch.Tensor,
    targets: torch.Tensor,
    iters: int = 10,
    smooth: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = probs.clamp(0.0, 1.0)
    targets = targets.float()
    sp = soft_skeletonize(probs, iters)
    sg = soft_skeletonize(targets, iters)

    tprec = (sp * targets).sum(dim=(2, 3))
    tprec = (tprec + eps) / (sp.sum(dim=(2, 3)) + eps)

    tsens = (sg * probs).sum(dim=(2, 3))
    tsens = (tsens + eps) / (sg.sum(dim=(2, 3)) + eps)

    cldice = (2 * tprec * tsens + smooth) / (tprec + tsens + smooth)
    return (1 - cldice).mean()


def tv_loss(probs: torch.Tensor) -> torch.Tensor:
    """Total variation regularization on prediction probabilities."""
    dx = probs[:, :, 1:, :] - probs[:, :, :-1, :]
    dy = probs[:, :, :, 1:] - probs[:, :, :, :-1]
    return dx.abs().mean() + dy.abs().mean()


def curvature_loss(probs: torch.Tensor) -> torch.Tensor:
    """Second-order smoothness via Laplacian penalty.

    Uses reflect padding instead of circular wrapping (torch.roll) to
    avoid phantom gradients at patch boundaries.
    """
    # Pad with reflect to avoid circular boundary artifacts
    padded = F.pad(probs, (1, 1, 1, 1), mode="reflect")
    lap = (
        -4.0 * probs
        + padded[:, :, :-2, 1:-1]   # up
        + padded[:, :, 2:, 1:-1]     # down
        + padded[:, :, 1:-1, :-2]    # left
        + padded[:, :, 1:-1, 2:]     # right
    )
    return lap.abs().mean()


def soft_dbc_loss(
    probs: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Soft differentiable boundary Dice loss.

    Boundary maps are approximated via soft morphological gradient
    (soft dilation - soft erosion), then matched with Dice.
    """
    probs = probs.clamp(0.0, 1.0)
    targets = targets.float().clamp(0.0, 1.0)
    probs_b = F.relu(_soft_dilate(probs) - _soft_erode(probs))
    targets_b = F.relu(_soft_dilate(targets) - _soft_erode(targets))
    return dice_loss(probs_b, targets_b, smooth=smooth, eps=eps)


def cbdice_loss(
    probs: torch.Tensor,
    targets: torch.Tensor,
    iters: int = 10,
    smooth: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Centerline-Boundary Dice loss (cbDice, MICCAI 2024).

    Extends clDice to penalize diameter inconsistency by weighting
    skeleton voxels by the local vessel radius (approximated via
    distance transform on the soft mask).
    """
    probs = probs.clamp(0.0, 1.0)
    targets = targets.float()

    # Soft skeletons
    sp = soft_skeletonize(probs, iters)
    sg = soft_skeletonize(targets, iters)

    # Approximate radius weighting via distance from boundary
    # Use soft erosion iterations as a proxy for distance transform
    dist_pred = probs.clone()
    dist_gt = targets.clone()
    for _ in range(3):
        dist_pred = _soft_erode(dist_pred)
        dist_gt = _soft_erode(dist_gt)
    # Radius weight: skeleton pixels weighted by how "deep" they are
    w_pred = (dist_pred + eps).clamp(0, 1)
    w_gt = (dist_gt + eps).clamp(0, 1)

    # Weighted skeleton precision and sensitivity
    tprec = (sp * targets * w_gt).sum(dim=(2, 3))
    tprec = (tprec + eps) / ((sp * w_gt).sum(dim=(2, 3)) + eps)

    tsens = (sg * probs * w_pred).sum(dim=(2, 3))
    tsens = (tsens + eps) / ((sg * w_pred).sum(dim=(2, 3)) + eps)

    cbdice = (2 * tprec * tsens + smooth) / (tprec + tsens + smooth)
    return (1 - cbdice).mean()


def skeleton_recall_loss(
    probs: torch.Tensor,
    targets: torch.Tensor,
    iters: int = 10,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Skeleton Recall Loss — efficient topology preservation.

    Penalizes missed skeleton pixels (false negatives on the
    centerline) which represent topological disconnections.
    >90% cheaper than persistence-based topology losses.
    """
    targets = targets.float()
    sg = soft_skeletonize(targets, iters)

    # Recall: what fraction of GT skeleton is covered by prediction?
    recall = (sg * probs).sum(dim=(2, 3)) / (sg.sum(dim=(2, 3)) + eps)
    return (1 - recall).mean()


def fractal_weighted_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    fractal_weight: torch.Tensor,
    pos_weight: float | None = None,
    alpha: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """BCE weighted by fractal complexity map (ECCV 2024 FFM).

    Regions with higher fractal dimension (branching, thin vessels)
    get higher loss weight, focusing learning on structurally
    complex areas.

    Args:
        logits: (B, 1, H, W) raw logits.
        targets: (B, 1, H, W) binary targets.
        fractal_weight: (B, 1, H, W) normalized fractal map [0,1].
        pos_weight: Optional positive class weight.
        alpha: Strength of fractal weighting (1.0 = full effect).
    """
    targets = targets.float()

    # Compute per-pixel weight: base=1 + alpha * fractal_complexity
    fw = fractal_weight.to(logits.device)
    if fw.shape[2:] != logits.shape[2:]:
        fw = F.interpolate(fw, size=logits.shape[2:], mode="bilinear", align_corners=False)
    pixel_weight = 1.0 + alpha * fw

    # Standard BCE
    if pos_weight is not None:
        pw = torch.tensor(float(pos_weight), device=logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw, reduction="none")
    else:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    weighted_bce = (bce * pixel_weight).mean()
    return weighted_bce


class CompositeLoss(torch.nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        fractal_weight: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_cfg = self.config
        from_logits = bool(loss_cfg.get("from_logits", True))

        probs = torch.sigmoid(logits) if from_logits else logits
        targets = targets.float()

        dice_cfg = loss_cfg.get("dice", {})
        bce_cfg = loss_cfg.get("bce", {})
        focal_cfg = loss_cfg.get("focal", {})
        cldice_cfg = loss_cfg.get("cldice", {})
        soft_dbc_cfg = loss_cfg.get("soft_dbc", {})
        cbdice_cfg = loss_cfg.get("cbdice", {})
        skel_recall_cfg = loss_cfg.get("skeleton_recall", {})
        fractal_bce_cfg = loss_cfg.get("fractal_bce", {})
        pde_cfg = loss_cfg.get("pde", {})
        tv_cfg = pde_cfg.get("tv", {})
        curv_cfg = pde_cfg.get("curvature", {})

        dice_enabled = bool(dice_cfg.get("enabled", True))
        bce_enabled = bool(bce_cfg.get("enabled", True))
        focal_enabled = bool(focal_cfg.get("enabled", False))
        cldice_enabled = bool(cldice_cfg.get("enabled", False))
        soft_dbc_enabled = bool(soft_dbc_cfg.get("enabled", False))
        cbdice_enabled = bool(cbdice_cfg.get("enabled", False))
        skel_recall_enabled = bool(skel_recall_cfg.get("enabled", False))
        fractal_bce_enabled = bool(fractal_bce_cfg.get("enabled", False))
        tv_enabled = bool(tv_cfg.get("enabled", False))
        curv_enabled = bool(curv_cfg.get("enabled", False))

        loss_dice = torch.tensor(0.0, device=logits.device)
        if dice_enabled:
            smooth = float(dice_cfg.get("smooth", 1.0))
            loss_dice = dice_loss(probs, targets, smooth=smooth)

        loss_bce = torch.tensor(0.0, device=logits.device)
        if bce_enabled:
            pos_weight = bce_cfg.get("pos_weight")
            pos_tensor = None
            if pos_weight is not None:
                pos_tensor = torch.tensor(float(pos_weight), device=logits.device)
            loss_bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_tensor)

        loss_focal = torch.tensor(0.0, device=logits.device)
        if focal_enabled:
            loss_focal = focal_loss(
                logits,
                targets,
                alpha=float(focal_cfg.get("alpha", 0.25)),
                gamma=float(focal_cfg.get("gamma", 2.0)),
                reduction=str(focal_cfg.get("reduction", "mean")),
            )

        loss_cldice = torch.tensor(0.0, device=logits.device)
        if cldice_enabled:
            loss_cldice = cldice_loss(
                probs,
                targets,
                iters=int(cldice_cfg.get("iter", 10)),
                smooth=float(cldice_cfg.get("smooth", 1.0)),
                eps=float(cldice_cfg.get("eps", 1e-6)),
            )

        loss_soft_dbc = torch.tensor(0.0, device=logits.device)
        if soft_dbc_enabled:
            loss_soft_dbc = soft_dbc_loss(
                probs,
                targets,
                smooth=float(soft_dbc_cfg.get("smooth", 1.0)),
                eps=float(soft_dbc_cfg.get("eps", 1e-6)),
            )

        loss_cbdice = torch.tensor(0.0, device=logits.device)
        if cbdice_enabled:
            loss_cbdice = cbdice_loss(
                probs,
                targets,
                iters=int(cbdice_cfg.get("iter", 10)),
                smooth=float(cbdice_cfg.get("smooth", 1.0)),
                eps=float(cbdice_cfg.get("eps", 1e-6)),
            )

        loss_skel_recall = torch.tensor(0.0, device=logits.device)
        if skel_recall_enabled:
            loss_skel_recall = skeleton_recall_loss(
                probs,
                targets,
                iters=int(skel_recall_cfg.get("iter", 10)),
                eps=float(skel_recall_cfg.get("eps", 1e-6)),
            )

        loss_fractal_bce = torch.tensor(0.0, device=logits.device)
        if fractal_bce_enabled and fractal_weight is not None:
            loss_fractal_bce = fractal_weighted_bce(
                logits,
                targets,
                fractal_weight,
                pos_weight=fractal_bce_cfg.get("pos_weight"),
                alpha=float(fractal_bce_cfg.get("alpha", 1.0)),
            )

        loss_tv = torch.tensor(0.0, device=logits.device)
        if tv_enabled:
            loss_tv = tv_loss(probs)

        loss_curvature = torch.tensor(0.0, device=logits.device)
        if curv_enabled:
            loss_curvature = curvature_loss(probs)

        total = torch.tensor(0.0, device=logits.device)
        if dice_enabled:
            total = total + float(dice_cfg.get("weight", 1.0)) * loss_dice
        if bce_enabled:
            total = total + float(bce_cfg.get("weight", 1.0)) * loss_bce
        if focal_enabled:
            total = total + float(focal_cfg.get("weight", 0.0)) * loss_focal
        if cldice_enabled:
            total = total + float(cldice_cfg.get("weight", 0.0)) * loss_cldice
        if soft_dbc_enabled:
            total = total + float(soft_dbc_cfg.get("weight", 0.0)) * loss_soft_dbc
        if cbdice_enabled:
            total = total + float(cbdice_cfg.get("weight", 0.0)) * loss_cbdice
        if skel_recall_enabled:
            total = total + float(skel_recall_cfg.get("weight", 0.0)) * loss_skel_recall
        if fractal_bce_enabled and fractal_weight is not None:
            total = total + float(fractal_bce_cfg.get("weight", 0.0)) * loss_fractal_bce
        if tv_enabled:
            total = total + float(tv_cfg.get("weight", 0.0)) * loss_tv
        if curv_enabled:
            total = total + float(curv_cfg.get("weight", 0.0)) * loss_curvature

        breakdown = {
            "loss_total": float(total.detach().item()),
            "loss_dice": float(loss_dice.detach().item()),
            "loss_bce": float(loss_bce.detach().item()),
            "loss_focal": float(loss_focal.detach().item()),
            "loss_cldice": float(loss_cldice.detach().item()),
            "loss_soft_dbc": float(loss_soft_dbc.detach().item()),
            "loss_cbdice": float(loss_cbdice.detach().item()),
            "loss_skel_recall": float(loss_skel_recall.detach().item()),
            "loss_fractal_bce": float(loss_fractal_bce.detach().item()),
            "loss_tv": float(loss_tv.detach().item()),
            "loss_curvature": float(loss_curvature.detach().item()),
        }

        return total, breakdown
