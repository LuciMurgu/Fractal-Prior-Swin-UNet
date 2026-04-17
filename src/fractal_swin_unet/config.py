"""Shared config loading and override utilities."""

from __future__ import annotations

import json
from typing import Any

import yaml


def load_config(path: str) -> dict[str, Any]:
    """Load a YAML config file and return a dict."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def parse_value(raw: str) -> Any:
    """Parse a CLI override value string to the appropriate Python type."""
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    try:
        return json.loads(raw)
    except Exception:
        return raw


def apply_override(cfg: dict[str, Any], override: str) -> None:
    """Apply a dotted key=value override to a nested config dict."""
    if "=" not in override:
        raise ValueError("Override must be in key=value format.")
    key, value = override.split("=", 1)
    keys = key.split(".")
    cursor = cfg
    for k in keys[:-1]:
        if k not in cursor or not isinstance(cursor[k], dict):
            cursor[k] = {}
        cursor = cursor[k]
    cursor[keys[-1]] = parse_value(value)
