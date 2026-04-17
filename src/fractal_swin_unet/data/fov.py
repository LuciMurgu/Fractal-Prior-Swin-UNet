"""FOV mask loading utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_fov_mask(path: str | Path, target_shape: tuple[int, int] | None = None) -> torch.Tensor:
    """Load FOV mask from .pt or image file and return bool mask."""

    mask_path = Path(path)
    if mask_path.suffix.lower() == ".pt":
        mask = torch.load(mask_path, map_location="cpu")
        if isinstance(mask, dict) and "mask" in mask:
            mask = mask["mask"]
    else:
        img = Image.open(mask_path).convert("L")
        mask = torch.from_numpy(np.array(img, dtype=np.float32))

    if mask.ndim == 3:
        mask = mask.squeeze(0)
    if mask.ndim != 2:
        raise ValueError("FOV mask must be 2D")
    mask = (mask > 0).float()

    if target_shape is not None and (mask.shape[0], mask.shape[1]) != target_shape:
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = torch.nn.functional.interpolate(mask, size=target_shape, mode="nearest")
        mask = mask.squeeze(0).squeeze(0)
    return mask.bool()
