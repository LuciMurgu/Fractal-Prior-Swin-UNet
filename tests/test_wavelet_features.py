from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fractal_swin_unet.data.hrf import HRFDataset
from fractal_swin_unet.data.wavelets import apply_wavelet_projections, wavelet_channel_count


def test_wavelet_projection_shape_and_determinism() -> None:
    image = torch.rand(3, 15, 17)
    cfg = {
        "enabled": True,
        "include_input": True,
        "projections": ["gray", "green"],
        "levels": 2,
        "bands": ["ll", "lh", "hl", "hh"],
    }

    out1 = apply_wavelet_projections(image, cfg)
    out2 = apply_wavelet_projections(image, cfg)

    assert out1.shape == (wavelet_channel_count(3, cfg), 15, 17)
    assert torch.allclose(out1, out2)


def test_hrf_dataset_wavelet_channels(tmp_path: Path) -> None:
    root = tmp_path
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[..., 1] = 128
    Image.fromarray(img).save(root / "img.png")
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(root / "mask.png")

    manifest = [{"id": "img", "image_path": "img.png", "mask_path": "mask.png"}]
    wavelet_cfg = {
        "enabled": True,
        "include_input": False,
        "projections": ["green"],
        "levels": 1,
        "bands": ["ll", "hh"],
    }
    dataset = HRFDataset(manifest, dataset_root=root, wavelet_config=wavelet_cfg)
    sample = dataset[0]

    assert sample["image"].shape == (2, 8, 8)
