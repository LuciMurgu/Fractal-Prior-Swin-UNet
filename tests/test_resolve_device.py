"""Device resolution."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from fractal_swin_unet.device import resolve_device


def test_resolve_device_cuda_prefers_cuda0_when_available() -> None:
    with patch("fractal_swin_unet.device.torch.cuda.is_available", return_value=True):
        d = resolve_device()
    assert str(d) == "cuda:0"


def test_resolve_device_cpu_when_no_cuda() -> None:
    with patch("fractal_swin_unet.device.torch.cuda.is_available", return_value=False):
        d = resolve_device()
    assert str(d) == "cpu"


def test_resolve_device_require_gpu_raises() -> None:
    with patch("fractal_swin_unet.device.torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="GPU required"):
            resolve_device(require_gpu=True)
