"""Run artifact utilities for reproducibility."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

from .manifest import save_manifest


def make_run_dir(base: str = "runs", run_id: str | None = None) -> Path:
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(base) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_resolved_config(run_dir: Path, config: Dict[str, Any]) -> None:
    path = run_dir / "resolved_config.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def write_env(run_dir: Path) -> None:
    lines = [f"python={sys.version}"]
    for pkg in ("torch", "yaml", "numpy"):
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "unknown")
            lines.append(f"{pkg}={version}")
        except Exception:
            lines.append(f"{pkg}=unavailable")
    (run_dir / "env.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_git_commit(run_dir: Path) -> None:
    commit = "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        commit = result.stdout.strip()
    except Exception:
        pass
    (run_dir / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")


def write_code_hash(run_dir: Path, src_root: str = "src/fractal_swin_unet") -> None:
    root = Path(src_root)
    digest = hashlib.sha256()
    py_files = sorted(p for p in root.rglob("*.py") if "__pycache__" not in str(p))
    for path in py_files:
        digest.update(path.read_bytes())
    (run_dir / "code_hash.txt").write_text(digest.hexdigest() + "\n", encoding="utf-8")


def write_manifest_and_hash(run_dir: Path, samples: Iterable[Dict[str, Any]]) -> None:
    manifest_path = run_dir / "manifest_used.jsonl"
    save_manifest(samples, manifest_path)
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (run_dir / "manifest_hash.txt").write_text(digest + "\n", encoding="utf-8")


def write_metrics(run_dir: Path, metrics: Dict[str, Any]) -> None:
    path = run_dir / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def write_threshold(run_dir: Path, payload: Dict[str, Any]) -> None:
    path = run_dir / "threshold.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_readme(run_dir: Path, command: str) -> None:
    path = run_dir / "README_RUN.md"
    path.write_text(f"Run command:\n\n{command}\n", encoding="utf-8")
