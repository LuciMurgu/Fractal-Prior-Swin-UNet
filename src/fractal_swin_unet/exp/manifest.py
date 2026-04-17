"""Manifest utilities for reproducible dataset splits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Dict, Any


REQUIRED_KEYS = {"id", "image_path"}


def _ensure_required(sample: Dict[str, Any]) -> None:
    missing = REQUIRED_KEYS - set(sample.keys())
    if missing:
        raise ValueError(f"Manifest sample missing required keys: {missing}")


def load_manifest(path: str | Path, dataset_root: str | Path | None = None) -> List[Dict[str, Any]]:
    """Load a JSONL manifest from disk.

    Args:
        path: Path to JSONL manifest.
        dataset_root: Optional root to resolve relative paths.

    Returns:
        List of sample dicts.
    """

    manifest_path = Path(path)
    samples: List[Dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            _ensure_required(sample)
            samples.append(sample)

    return normalize_paths(samples, dataset_root)


def save_manifest(samples: Iterable[Dict[str, Any]], path: str | Path) -> None:
    """Save a list of samples to JSONL."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            _ensure_required(sample)
            handle.write(json.dumps(sample, ensure_ascii=True) + "\n")


def normalize_paths(
    samples: Iterable[Dict[str, Any]],
    dataset_root: str | Path | None,
) -> List[Dict[str, Any]]:
    """Normalize relative paths in samples.

    Args:
        samples: Sample list.
        dataset_root: Optional root to prefix relative paths.

    Returns:
        New list of samples with normalized paths.
    """

    if dataset_root is None:
        return [dict(sample) for sample in samples]

    root = Path(dataset_root)
    normalized: List[Dict[str, Any]] = []
    for sample in samples:
        entry = dict(sample)
        image_path = Path(entry["image_path"])
        if not image_path.is_absolute():
            entry["image_path"] = str(root / image_path)
        if "mask_path" in entry and entry["mask_path"]:
            mask_path = Path(entry["mask_path"])
            if not mask_path.is_absolute():
                entry["mask_path"] = str(root / mask_path)
        # C4: Also normalize fov_mask_path to prevent broken FOV loading
        if "fov_mask_path" in entry and entry["fov_mask_path"]:
            fov_path = Path(entry["fov_mask_path"])
            if not fov_path.is_absolute():
                entry["fov_mask_path"] = str(root / fov_path)
        normalized.append(entry)
    return normalized
