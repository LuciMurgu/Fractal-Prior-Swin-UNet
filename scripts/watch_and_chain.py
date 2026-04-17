#!/usr/bin/env python3
"""Watcher: Wait for baseline_200ep to finish → launch ablation chain.

Usage:
    PYTHONUNBUFFERED=1 nohup .venv/bin/python3 scripts/watch_and_chain.py > runs/watch_chain.log 2>&1 &
"""
import subprocess
import time
import json
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
VENV_PYTHON = str(PROJECT / ".venv" / "bin" / "python3")
BASELINE_METRICS = PROJECT / "runs" / "baseline_200ep" / "metrics.json"
POLL_INTERVAL = 300  # Check every 5 minutes


def baseline_still_running() -> bool:
    """Check if the baseline training process is still alive."""
    result = subprocess.run(
        ["pgrep", "-f", "baseline_200ep"],
        capture_output=True,
    )
    return result.returncode == 0


def main():
    os.chdir(str(PROJECT))
    print("🔭 WATCHER: Waiting for baseline_200ep to finish...", flush=True)
    print(f"   Polling every {POLL_INTERVAL}s", flush=True)

    while True:
        # Check if metrics.json exists (training complete)
        if BASELINE_METRICS.exists():
            m = json.loads(BASELINE_METRICS.read_text())
            dice = m.get("best_full_eval_dice", m.get("val_dice", "?"))
            print(f"\n✅ Baseline finished! Dice = {dice}", flush=True)
            break

        # Check if process is still running
        if not baseline_still_running():
            print("\n⚠ Baseline process not found. Checking for metrics...", flush=True)
            time.sleep(30)  # Give filesystem time to flush
            if BASELINE_METRICS.exists():
                m = json.loads(BASELINE_METRICS.read_text())
                dice = m.get("best_full_eval_dice", m.get("val_dice", "?"))
                print(f"✅ Baseline finished! Dice = {dice}", flush=True)
                break
            else:
                print("⚠ Baseline crashed without metrics. Launching ablation anyway.", flush=True)
                break

        # Still running — report and wait
        # Check latest epoch from log
        log_path = PROJECT / "runs" / "baseline_200ep.log"
        if log_path.exists():
            lines = log_path.read_text().strip().split("\n")
            if lines:
                last_line = lines[-1]
                epoch_info = last_line.split(" ")[1] if "Epoch" in last_line else "?"
                print(f"  ⏳ Baseline still training (epoch {epoch_info}/200)...", flush=True)

        time.sleep(POLL_INTERVAL)

    # Clean GPU memory
    print("\n🧹 Cleaning GPU memory...", flush=True)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    time.sleep(10)

    # Launch ablation chain
    print("\n🚀 LAUNCHING ABLATION CHAIN (OPT-F → OPT-G)...", flush=True)
    result = subprocess.run(
        [VENV_PYTHON, "scripts/overnight_ablation_chain.py"],
        cwd=str(PROJECT),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    if result.returncode == 0:
        print("\n🎉 Ablation chain completed successfully!", flush=True)
    else:
        print(f"\n⚠ Ablation chain finished with rc={result.returncode}", flush=True)


if __name__ == "__main__":
    main()
