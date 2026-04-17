from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fractal_swin_unet.data.hrf import HRFDataset


def test_hrf_mask_binarization(tmp_path: Path) -> None:
    root = tmp_path
    img_path = root / "img.png"
    mask_path = root / "mask.png"
    fov_path = root / "fov.png"

    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(img_path)
    Image.fromarray(np.array([[0, 10, 0, 255]] * 4, dtype=np.uint8)).save(mask_path)
    Image.fromarray(np.array([[0, 0, 255, 255]] * 4, dtype=np.uint8)).save(fov_path)

    manifest = [
        {
            "id": "img",
            "image_path": "img.png",
            "mask_path": "mask.png",
            "fov_mask_path": "fov.png",
        }
    ]

    dataset = HRFDataset(manifest, dataset_root=root, return_fov=True)
    sample = dataset[0]

    mask = sample["mask"]
    assert mask.unique().tolist() == [0.0, 1.0]

    fov = sample["fov"]
    assert fov.dtype == torch.bool
