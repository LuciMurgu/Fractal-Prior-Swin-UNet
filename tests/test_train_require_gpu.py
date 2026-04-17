"""GPU-only training flag."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from fractal_swin_unet.train import train


def test_require_gpu_raises_when_cuda_unavailable() -> None:
    cfg = {
        "seed": 0,
        "train": {"epochs": 0},
        "data": {"manifest_path": "manifests/hrf.jsonl", "dataset_root": "/tmp", "dataset": "hrf"},
        "model": {"name": "baseline"},
    }
    run_dir = Path("/tmp/fractal_train_require_gpu_test")
    run_dir.mkdir(parents=True, exist_ok=True)
    with patch("fractal_swin_unet.device.torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="GPU required"):
            train(cfg, use_synth_data=True, overfit_one_batch=False, run_dir=run_dir, require_gpu=True)
