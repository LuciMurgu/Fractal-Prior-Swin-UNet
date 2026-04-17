"""Device selection for training and inference."""

from __future__ import annotations

import torch


def resolve_device(*, require_gpu: bool = False) -> torch.device:
    """Use the first available CUDA device (``cuda:0``) when PyTorch sees a GPU.

    Respects ``CUDA_VISIBLE_DEVICES``: the visible GPU is always index ``0`` from
    PyTorch's perspective.
    """
    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU required but CUDA is not available "
            "(no GPU driver, PyTorch CUDA build, or visible CUDA device)."
        )
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")
