"""Meijering neuriteness filter for thin tubular structure enhancement.

Similar to Frangi but uses a modified eigenvalue criterion optimized for
neurite/capillary-like structures. Better sensitivity to thin vessels.

Reference: Meijering et al., "Design and validation of a tool for neurite
tracing and analysis in fluorescence microscopy images" (2004)
"""

from __future__ import annotations

from typing import Sequence

import torch

from .frangi import _gaussian_blur, _hessian_2d, _eigenvalues_2d


def meijering_neuriteness(
    image: torch.Tensor,
    sigmas: Sequence[float] = (0.5, 1.0, 2.0),
    black_ridges: bool = True,
) -> torch.Tensor:
    """Compute Meijering neuriteness filter response.

    The Meijering filter modifies the eigenvalue criterion to be more
    sensitive to thin elongated structures (capillaries) compared to Frangi.

    Neuriteness = max(0, λ_modified)  where:
        λ_modified = λ2 + α·λ1  (α = 1/3 for 2D)

    Args:
        image: Input tensor of shape (C, H, W) or (B, C, H, W).
        sigmas: Scales for multi-scale analysis. Smaller sigmas than Frangi
            to capture fine capillaries.
        black_ridges: If True, detect dark vessels on bright background.

    Returns:
        Neuriteness map of shape (1, H, W) or (B, 1, H, W), in [0, 1].
    """
    squeeze = False
    if image.ndim == 3:
        image = image.unsqueeze(0)
        squeeze = True

    # Convert to grayscale
    if image.shape[1] == 3:
        gray = 0.2989 * image[:, 0:1] + 0.5870 * image[:, 1:2] + 0.1140 * image[:, 2:3]
    elif image.shape[1] == 1:
        gray = image
    else:
        gray = image.mean(dim=1, keepdim=True)

    alpha = 1.0 / 3.0  # Meijering constant for 2D

    max_neuriteness = torch.zeros_like(gray)

    for sigma in sigmas:
        Ixx, Ixy, Iyy = _hessian_2d(gray, sigma)
        lam1, lam2 = _eigenvalues_2d(Ixx, Ixy, Iyy)

        # Meijering criterion: modified eigenvalue
        # For bright ridges: most negative eigenvalue combination
        # λ_mod = λ2 + α·λ1  (where |λ1| ≤ |λ2|)
        lam_mod = lam2 + alpha * lam1

        if black_ridges:
            # Dark vessels on bright background → positive eigenvalues
            neuriteness = torch.clamp(lam_mod, min=0.0)
        else:
            # Bright vessels → take negative of the combination
            neuriteness = torch.clamp(-lam_mod, min=0.0)

        # Normalize per-scale by λ2² for scale invariance
        scale_norm = lam2 ** 2 + 1e-8
        neuriteness = neuriteness ** 2 / scale_norm

        max_neuriteness = torch.maximum(max_neuriteness, neuriteness)

    # Normalize to [0, 1]
    vmax = max_neuriteness.amax(dim=(-2, -1), keepdim=True)
    result = max_neuriteness / (vmax + 1e-8)

    if squeeze:
        result = result.squeeze(0)
    return result
