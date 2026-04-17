#!/usr/bin/env python3
"""Chain: Wait for OPT-D v2 → Run Baseline 200ep → Eval both.

This is the critical control experiment. It answers:
  "Is the improvement from the fractal prior, or just from more training?"

Expected result if fractal prior works:
  fractal@200ep > baseline@200ep  →  Prior adds value beyond training
  
If baseline@200ep ≈ fractal@200ep:
  The improvement was just from more epochs (bad for the paper)
"""
import subprocess
import sys
import os
import json
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
VENV_PYTHON = str(PROJECT / ".venv" / "bin" / "python3")


def run(cmd, desc, timeout=None):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}\n", flush=True)
    result = subprocess.run(cmd, cwd=str(PROJECT), timeout=timeout)
    if result.returncode != 0:
        print(f"  ⚠ FAILED (rc={result.returncode})", flush=True)
    return result.returncode


def wait_for_optd_v2():
    """Wait for OPT-D v2 to finish (check for metrics.json)."""
    optd_metrics = PROJECT / "runs" / "opt_D_v2_200ep" / "metrics.json"
    
    # Check if process is still running
    while True:
        result = subprocess.run(
            ["pgrep", "-f", "opt_D_v2_200ep"],
            capture_output=True
        )
        if result.returncode != 0:
            # Process not found — either finished or never started
            break
        
        print(f"  ⏳ OPT-D v2 still running... waiting 5 min", flush=True)
        time.sleep(300)  # Check every 5 min
    
    if optd_metrics.exists():
        m = json.loads(optd_metrics.read_text())
        dice = m.get("best_full_eval_dice", "?")
        print(f"  ✅ OPT-D v2 finished! Dice = {dice}", flush=True)
    else:
        print(f"  ⚠ OPT-D v2 process ended but no metrics found", flush=True)


def main():
    os.chdir(str(PROJECT))

    # --- Phase 0: Wait for OPT-D v2 ---
    print("\n⏳ PHASE 0: WAITING FOR OPT-D v2 TO FINISH...", flush=True)
    wait_for_optd_v2()

    # Clean GPU memory
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Phase 1: Train Baseline 200ep ---
    print("\n🔬 PHASE 1: BASELINE 200ep (CONTROL EXPERIMENT)", flush=True)
    
    BASELINE_CONFIG = "configs/optim/BASELINE_200ep.yaml"
    BASELINE_RUN = "baseline_200ep"
    
    # Can resume from existing baseline checkpoint if available
    baseline_resume = PROJECT / "runs" / "xdataset_fives_baseline" / "checkpoint_best.pt"
    
    train_cmd = [
        VENV_PYTHON, "-m", "fractal_swin_unet.train",
        "--config", BASELINE_CONFIG,
        "--run_id", BASELINE_RUN,
    ]
    if baseline_resume.exists():
        train_cmd += ["--resume_from", str(baseline_resume)]
        print(f"  ✓ Resuming from {baseline_resume}", flush=True)
    
    rc = run(train_cmd, "Training Baseline 200ep (no fractal prior)")
    if rc != 0:
        print("Baseline training failed!", flush=True)
        sys.exit(1)

    # --- Phase 2: Eval Baseline on FIVES test ---
    print("\n📊 PHASE 2: EVAL BASELINE ON FIVES", flush=True)
    run([
        VENV_PYTHON, "-m", "fractal_swin_unet.eval",
        "--config", BASELINE_CONFIG,
        "--run_id", BASELINE_RUN,
    ], "Baseline eval with threshold sweep + clDice", timeout=7200)

    # --- Phase 3: Cross-dataset eval on Baseline ---
    print("\n🌍 PHASE 3: BASELINE CROSS-DATASET EVAL", flush=True)
    datasets = {
        "DRIVE": ("manifests/drive.jsonl", "/home/lucian/medical-imaging-ai/Retina Datasets/Full retina dataset/DRIVE"),
        "CHASE_DB1": ("manifests/chase_db1.jsonl", "/home/lucian/medical-imaging-ai/Retina Datasets/Full retina dataset/CHASE_DB1"),
        "STARE": ("manifests/stare.jsonl", "/home/lucian/medical-imaging-ai/Retina Datasets/Full retina dataset/STARE"),
        "HRF": ("manifests/hrf_full.jsonl", "/home/lucian/medical-imaging-ai/Retina Datasets/Full retina dataset/HRF"),
    }
    for ds_name, (manifest, ds_root) in datasets.items():
        if not Path(manifest).exists() or not Path(ds_root).exists():
            print(f"  ⚠ Skipping {ds_name}", flush=True)
            continue
        run([
            VENV_PYTHON, "-m", "fractal_swin_unet.eval",
            "--config", BASELINE_CONFIG,
            "--run_id", BASELINE_RUN,
            "--set", f"data.manifest_path={manifest}",
            "--set", f"data.dataset_root={ds_root}",
        ], f"Baseline cross-dataset eval on {ds_name}", timeout=7200)

    # --- Summary ---
    print("\n" + "=" * 60, flush=True)
    print("  FINAL COMPARISON: FRACTAL vs BASELINE @ 200 EPOCHS", flush=True)
    print("=" * 60, flush=True)
    
    for run_id, label in [("opt_D_v2_200ep", "FRACTAL"), ("baseline_200ep", "BASELINE")]:
        mp = PROJECT / "runs" / run_id / "metrics.json"
        if mp.exists():
            m = json.loads(mp.read_text())
            dice = m.get("best_full_eval_dice", "?")
            cldice = m.get("test_cldice_tau_star", m.get("best_cldice", "?"))
            tau = m.get("best_full_eval_tau_star", "?")
            epoch = m.get("best_full_eval_epoch", "?")
            if isinstance(dice, float): dice = f"{dice:.4f}"
            if isinstance(cldice, float): cldice = f"{cldice:.4f}"
            print(f"  {label:<12} Dice={dice}  clDice={cldice}  τ*={tau}  ep={epoch}")
        else:
            print(f"  {label:<12} NO METRICS FOUND")
    
    print("\nDONE! 🎉", flush=True)


if __name__ == "__main__":
    main()
