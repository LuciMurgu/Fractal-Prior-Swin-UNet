"""Experiment matrix loading and overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_matrix(path: str | Path) -> Dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "matrix" not in data or "experiments" not in data:
        raise ValueError("Matrix config must contain 'matrix' and 'experiments'.")
    if not isinstance(data["experiments"], list) or not data["experiments"]:
        raise ValueError("Matrix config 'experiments' must be a non-empty list.")
    for exp in data["experiments"]:
        if "name" not in exp or "config" not in exp:
            raise ValueError("Each experiment must include 'name' and 'config'.")
    return data


def deep_update(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if "." in key:
            parts = key.split(".")
            cursor = cfg
            for part in parts[:-1]:
                if part not in cursor or not isinstance(cursor[part], dict):
                    cursor[part] = {}
                cursor = cursor[part]
            cursor[parts[-1]] = value
        else:
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key] = deep_update(cfg[key], value)
            else:
                cfg[key] = value
    return cfg
