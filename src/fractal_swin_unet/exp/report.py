"""Report aggregation utilities for experiment matrices."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict

import yaml


def collect_run_metrics(run_dir: Path) -> Dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    threshold_path = run_dir / "threshold.json"
    config_path = run_dir / "resolved_config.yaml"
    manifest_hash_path = run_dir / "manifest_hash.txt"
    code_hash_path = run_dir / "code_hash.txt"

    metrics = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    if threshold_path.exists():
        threshold = json.loads(threshold_path.read_text(encoding="utf-8"))
        metrics["best_dice_val"] = threshold.get("best_dice")
        metrics["dice_at_0_5_val"] = threshold.get("dice_at_0_5")
        metrics["best_tau"] = threshold.get("tau_star")

    if config_path.exists():
        metrics["config"] = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    if manifest_hash_path.exists():
        metrics["manifest_hash"] = manifest_hash_path.read_text(encoding="utf-8").strip()
    if code_hash_path.exists():
        metrics["code_hash"] = code_hash_path.read_text(encoding="utf-8").strip()

    metrics["run_id"] = run_dir.name
    return metrics


def generate_summary(run_dirs: Dict[str, Path], reports_dir: Path, matrix_name: str) -> Dict[str, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    runs_map = {}

    for exp_name, run_dir in run_dirs.items():
        metrics = collect_run_metrics(run_dir)
        row = {
            "experiment": exp_name,
            "run_id": metrics.get("run_id"),
            "best_dice_val": metrics.get("best_dice_val"),
            "dice_at_0_5_val": metrics.get("dice_at_0_5_val"),
            "best_tau": metrics.get("best_tau"),
            "train_dice": metrics.get("train_dice"),
            "val_dice": metrics.get("val_dice"),
            "code_hash": metrics.get("code_hash"),
            "manifest_hash": metrics.get("manifest_hash"),
        }
        summary_rows.append(row)
        runs_map[exp_name] = {"run_id": run_dir.name, "run_dir": str(run_dir)}

    summary_rows.sort(key=lambda r: r.get("best_dice_val") or 0, reverse=True)

    csv_path = reports_dir / f"{matrix_name}_summary.csv"
    md_path = reports_dir / f"{matrix_name}_summary.md"
    json_path = reports_dir / f"{matrix_name}_runs.json"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with md_path.open("w", encoding="utf-8") as handle:
        headers = list(summary_rows[0].keys())
        handle.write("|" + "|".join(headers) + "|\n")
        handle.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in summary_rows:
            handle.write("|" + "|".join([str(row.get(h, "")) for h in headers]) + "|\n")

    json_path.write_text(json.dumps(runs_map, indent=2), encoding="utf-8")

    return {"csv": csv_path, "md": md_path, "json": json_path}
