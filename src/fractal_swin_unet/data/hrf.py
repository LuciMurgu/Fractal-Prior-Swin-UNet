"""HRF dataset adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image

from .wavelets import apply_wavelet_projections


def _load_image(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1)


def _load_mask(path: Path) -> torch.Tensor:
    mask = Image.open(path).convert("L")
    arr = np.array(mask, dtype=np.float32)
    return torch.from_numpy(arr)


def _normalize_image(image: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "imagenet":
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (image / 255.0 - mean) / std
    return image / 255.0


def _apply_clahe(
    image: torch.Tensor,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> torch.Tensor:
    """Apply CLAHE to each channel of an image tensor (C, H, W) in [0, 255] range.

    Uses OpenCV if available, otherwise falls back to a pure-PIL approximation
    (histogram equalization on each channel).
    """
    try:
        import cv2

        arr = image.numpy().transpose(1, 2, 0).astype(np.uint8)
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_grid_size, tile_grid_size),
        )
        channels = [clahe.apply(arr[:, :, c]) for c in range(arr.shape[2])]
        enhanced = np.stack(channels, axis=2).astype(np.float32)
        return torch.from_numpy(enhanced).permute(2, 0, 1)
    except ImportError:
        # Fallback: simple per-channel histogram equalization via PIL
        from PIL import ImageOps

        result_channels = []
        for c in range(image.shape[0]):
            channel_np = image[c].numpy().astype(np.uint8)
            pil_ch = Image.fromarray(channel_np, mode="L")
            eq_ch = ImageOps.equalize(pil_ch)
            result_channels.append(np.array(eq_ch, dtype=np.float32))
        return torch.from_numpy(np.stack(result_channels, axis=0))


class HRFDataset:
    """Dataset that returns full HRF images and masks."""

    def __init__(
        self,
        manifest: List[Dict[str, object]],
        dataset_root: str | Path,
        use_green_channel: bool = False,
        normalize: str = "0_1",
        return_fov: bool = True,
        wavelet_config: Dict[str, object] | None = None,
        clahe_config: Dict[str, Any] | None = None,
    ) -> None:
        self.manifest = manifest
        self.root = Path(dataset_root)
        self.use_green = use_green_channel
        self.normalize = normalize
        self.return_fov = return_fov
        self.wavelet_config = wavelet_config or {}
        self.clahe_config = clahe_config or {}

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.manifest[idx]
        image_path = Path(str(rec["image_path"]))
        if not image_path.is_absolute():
            image_path = self.root / image_path
        mask_path = Path(str(rec["mask_path"]))
        if not mask_path.is_absolute():
            mask_path = self.root / mask_path
        fov_path = None
        if self.return_fov and rec.get("fov_mask_path"):
            fov_path = Path(str(rec.get("fov_mask_path")))
            if not fov_path.is_absolute():
                fov_path = self.root / fov_path

        image = _load_image(image_path)
        mask = _load_mask(mask_path)
        mask = (mask > 0).float().unsqueeze(0)

        # --- CLAHE preprocessing (before normalization, on raw uint8-range tensor) ---
        if bool(self.clahe_config.get("enabled", False)):
            clip_limit = float(self.clahe_config.get("clip_limit", 2.0))
            tile_size = int(self.clahe_config.get("tile_grid_size", 8))
            image = _apply_clahe(image, clip_limit=clip_limit, tile_grid_size=tile_size)

        if self.use_green:
            image = image[1:2]
        if self.normalize == "0_1":
            image = _normalize_image(image, "0_1")
        elif self.normalize == "imagenet":
            image = _normalize_image(image, "imagenet")
        image = apply_wavelet_projections(image, self.wavelet_config)

        sample = {"image": image, "mask": mask, "id": rec.get("id", str(idx))}
        if self.return_fov and fov_path is not None:
            fov = _load_mask(fov_path)
            sample["fov"] = (fov > 0)
        return sample
