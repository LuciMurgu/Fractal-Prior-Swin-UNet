"""Disk cache utilities for fractal prior maps."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import torch


def make_cache_key(sample_id: str, engine_name: str, params: Dict[str, Any]) -> str:
    payload = {
        "sample_id": sample_id,
        "engine": engine_name,
        "params": params,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cache_path(cache_dir: str | Path, cache_key: str) -> Path:
    return Path(cache_dir) / f"{cache_key}.pt"


def metadata_path(cache_dir: str | Path, cache_key: str) -> Path:
    return Path(cache_dir) / f"{cache_key}.json"


def load_cached_map(path: str | Path) -> torch.Tensor:
    tensor = torch.load(Path(path), map_location="cpu")
    if isinstance(tensor, dict) and "map" in tensor:
        tensor = tensor["map"]
    return tensor.float()


def save_cached_map(path: str | Path, tensor: torch.Tensor, dtype: str = "float16") -> None:
    out = tensor.detach().cpu()
    if dtype == "float16":
        out = out.half()
    elif dtype == "float32":
        out = out.float()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, Path(path))


def save_metadata(path: str | Path, metadata: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
