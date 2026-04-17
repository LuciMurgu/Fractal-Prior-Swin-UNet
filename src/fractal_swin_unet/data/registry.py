"""Dataset registry for project datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .hrf import HRFDataset
from ..exp.manifest import load_manifest


def build_dataset(cfg: Dict[str, Any]) -> HRFDataset:
    data_cfg = cfg.get("data", {})
    dataset_name = data_cfg.get("dataset")
    if dataset_name != "hrf":
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    manifest_path = data_cfg.get("manifest_path")
    dataset_root = data_cfg.get("dataset_root")
    if not manifest_path or not dataset_root:
        raise ValueError("HRF dataset requires manifest_path and dataset_root")

    samples = load_manifest(manifest_path, dataset_root=dataset_root)
    return HRFDataset(
        samples,
        dataset_root=dataset_root,
        use_green_channel=bool(data_cfg.get("use_green_channel", False)),
        normalize=str(data_cfg.get("normalize", "0_1")),
        return_fov=bool(data_cfg.get("return_fov", True)),
        wavelet_config=data_cfg.get("wavelet", {}),
        clahe_config=data_cfg.get("clahe", {}),
    )
