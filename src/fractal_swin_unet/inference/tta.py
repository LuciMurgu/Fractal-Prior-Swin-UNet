"""D4 Test-Time Augmentation for segmentation models.

Applies the 8 elements of the dihedral group D4 (identity + 3 rotations × 
{no flip, hflip}) and optionally multi-scale inference, then averages
sigmoid probabilities across all variants.
"""

from __future__ import annotations

from typing import Callable, Tuple

import torch


def _d4_transforms() -> list[Tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]]:
    """Return list of (forward_transform, inverse_transform) pairs for D4 group.
    
    Each operates on (B, C, H, W) tensors.
    """
    transforms = []
    for k in range(4):  # 0°, 90°, 180°, 270°
        for flip in [False, True]:  # no flip, hflip
            def fwd(x: torch.Tensor, _k: int = k, _f: bool = flip) -> torch.Tensor:
                if _f:
                    x = torch.flip(x, dims=[-1])
                if _k:
                    x = torch.rot90(x, k=_k, dims=[-2, -1])
                return x

            def inv(x: torch.Tensor, _k: int = k, _f: bool = flip) -> torch.Tensor:
                if _k:
                    x = torch.rot90(x, k=4 - _k, dims=[-2, -1])
                if _f:
                    x = torch.flip(x, dims=[-1])
                return x

            transforms.append((fwd, inv))
    return transforms


def tta_predict_proba(
    predict_fn: Callable[[torch.Tensor], torch.Tensor],
    image: torch.Tensor,
    use_d4: bool = True,
    scales: list[float] | None = None,
) -> torch.Tensor:
    """Run TTA prediction and return averaged probabilities.

    Args:
        predict_fn: Function that takes (B, C, H, W) image tensor and returns
            (B, 1, H, W) probability tensor (after sigmoid).
        image: Input image tensor of shape (B, C, H, W).
        use_d4: Whether to apply D4 geometric augmentations.
        scales: Optional list of scale factors for multi-scale TTA.
            E.g., [0.85, 1.0, 1.15]. Default is [1.0] (no scaling).

    Returns:
        Averaged probability tensor of shape (B, 1, H, W).
    """
    if scales is None:
        scales = [1.0]

    _, _, h, w = image.shape
    transforms = _d4_transforms() if use_d4 else [(lambda x: x, lambda x: x)]

    prob_sum = torch.zeros(image.shape[0], 1, h, w, device=image.device, dtype=image.dtype)
    count = 0

    for scale in scales:
        if scale != 1.0:
            sh = int(round(h * scale))
            sw = int(round(w * scale))
            # Ensure even dimensions for rotation compatibility
            sh = sh if sh % 2 == 0 else sh + 1
            sw = sw if sw % 2 == 0 else sw + 1
            scaled_image = torch.nn.functional.interpolate(
                image, size=(sh, sw), mode="bilinear", align_corners=False,
            )
        else:
            scaled_image = image

        for fwd, inv in transforms:
            augmented = fwd(scaled_image)

            with torch.no_grad():
                probs = predict_fn(augmented)

            # Inverse transform the predictions
            probs = inv(probs)

            # Resize back to original resolution if scaled
            if scale != 1.0:
                probs = torch.nn.functional.interpolate(
                    probs, size=(h, w), mode="bilinear", align_corners=False,
                )

            prob_sum += probs
            count += 1

    return prob_sum / count
