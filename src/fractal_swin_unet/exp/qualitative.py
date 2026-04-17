"""Qualitative panel export utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image


def export_panels(run_dir: Path, out_dir: Path, exp_name: str, limit: int = 3) -> None:
    try:
        preds_dir = run_dir / "preds"
        if not preds_dir.exists():
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        pred_files = sorted(preds_dir.glob("pred_*.png"))[:limit]
        for idx, pred_path in enumerate(pred_files):
            pred = Image.open(pred_path).convert("L")
            panel = Image.new("L", pred.size)
            panel.paste(pred)
            panel.save(out_dir / f"{exp_name}_{idx}.png")
    except Exception:
        return
