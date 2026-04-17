"""Experiment matrix runner."""

from __future__ import annotations

import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from .matrix import deep_update
from .report import collect_run_metrics, generate_summary
from .qualitative import export_panels
from ..train import train as train_fn
from ..eval import evaluate as eval_fn
from ..infer import infer as infer_fn
from ..exp.run_artifacts import make_run_dir, write_resolved_config, write_readme


def resolve_trained_checkpoint(run_dir: Path) -> str | None:
    """Prefer ``checkpoint_best.pt`` after training when present, else ``checkpoint.pt``."""
    best = run_dir / "checkpoint_best.pt"
    final = run_dir / "checkpoint.pt"
    if best.exists():
        return str(best)
    if final.exists():
        return str(final)
    return None


def _load_config(path: str) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _run_id(matrix_name: str, exp_name: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = exp_name.replace(" ", "_")
    return f"{matrix_name}_{safe_name}_{ts}"


def _cleanup_gpu() -> None:
    """Force-release GPU memory between experiments."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _find_existing_run(base_out_dir: str, matrix_name: str, exp_name: str) -> Path | None:
    """Find an existing completed run dir for this experiment (has metrics.json)."""
    safe_name = exp_name.replace(" ", "_")
    prefix = f"{matrix_name}_{safe_name}_"
    base = Path(base_out_dir)
    if not base.exists():
        return None
    for d in sorted(base.iterdir(), reverse=True):
        if d.is_dir() and d.name.startswith(prefix) and (d / "metrics.json").exists():
            return d
    return None


def run_matrix(matrix_path: str) -> Dict[str, Any]:
    from .matrix import load_matrix

    data = load_matrix(matrix_path)
    matrix = data["matrix"]
    experiments = data["experiments"]

    matrix_name = matrix.get("name", "matrix")
    base_out_dir = matrix.get("base_out_dir", "runs")
    reports_dir = Path(matrix.get("reports_dir", "reports"))
    use_synth = bool(matrix.get("use_synth_data", True))
    steps = matrix.get("steps", {"train": True, "eval": True, "infer": True})
    eval_split = matrix.get("eval_split", "val")
    infer_samples = int(matrix.get("infer_samples_per_run", 3))

    run_map: Dict[str, str] = {}
    run_dirs: Dict[str, Path] = {}

    for exp in experiments:
        # --- Resume-aware: skip completed experiments ---
        existing = _find_existing_run(base_out_dir, matrix_name, exp["name"])
        if existing is not None:
            print(f"SKIP {exp['name']}: already completed at {existing}")
            run_map[exp["name"]] = existing.name
            run_dirs[exp["name"]] = existing
            continue

        cfg = _load_config(exp["config"])
        overrides = exp.get("overrides", {})
        cfg = deep_update(cfg, overrides)

        run_id = _run_id(matrix_name, exp["name"])
        run_dir = make_run_dir(base=base_out_dir, run_id=run_id)
        cfg.setdefault("run", {})["id"] = run_dir.name
        cfg["run"]["base"] = str(run_dir.parent)

        write_resolved_config(run_dir, cfg)
        write_readme(run_dir, f"matrix {matrix_name} experiment {exp['name']}")

        trained_ckpt: str | None = None
        if steps.get("train", True):
            train_fn(cfg, use_synth_data=use_synth, overfit_one_batch=False, run_dir=run_dir)
            trained_ckpt = resolve_trained_checkpoint(run_dir)
            if trained_ckpt is None:
                raise FileNotFoundError(f"Training did not write a checkpoint under {run_dir}")
        else:
            trained_ckpt = resolve_trained_checkpoint(run_dir)

        if steps.get("eval", True):
            eval_fn(cfg, use_synth_data=use_synth, checkpoint=trained_ckpt, run_dir=run_dir)
        if steps.get("infer", True):
            infer_fn(cfg, use_synth_data=use_synth, checkpoint=trained_ckpt, run_dir=run_dir, tau=0.5)

        run_map[exp["name"]] = run_id
        run_dirs[exp["name"]] = run_dir

        export_panels(run_dir, reports_dir / f"{matrix_name}_qual", exp["name"], limit=infer_samples)

        # --- Memory cleanup between experiments ---
        print(f"Cleanup after {exp['name']}...")
        _cleanup_gpu()

    reports = generate_summary(run_dirs, reports_dir, matrix_name)
    reports["run_map"] = run_map
    return reports
