#!/usr/bin/env python3
"""Launch OPT-D v2 (200 epochs) + full post-training eval.

Resumes from OPT-D checkpoint (epoch 100) and continues to 200.
After training: computes Dice, clDice, and cross-dataset metrics.
"""
import subprocess
import sys
import os
import json
import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
VENV_PYTHON = str(PROJECT / ".venv" / "bin" / "python3")

CONFIG = "configs/optim/OPT_D_v2_200ep.yaml"
RUN_ID = "opt_D_v2_200ep"
RUN_DIR = PROJECT / "runs" / RUN_ID

RESUME_CKPT = PROJECT / "runs" / "opt_D_longer" / "checkpoint_best.pt"


def run(cmd, desc, timeout=None):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}\n", flush=True)
    result = subprocess.run(cmd, cwd=str(PROJECT), timeout=timeout)
    if result.returncode != 0:
        print(f"  ⚠ FAILED (rc={result.returncode})", flush=True)
    return result.returncode


def main():
    os.chdir(str(PROJECT))

    # --- Phase 1: Train 200 epochs (resume from OPT-D ep 100) ---
    print("\n🔥 PHASE 1: OPT-D v2 (200 EPOCHS)", flush=True)

    train_cmd = [
        VENV_PYTHON, "-m", "fractal_swin_unet.train",
        "--config", CONFIG,
        "--run_id", RUN_ID,
    ]

    if RESUME_CKPT.exists():
        train_cmd += ["--resume_from", str(RESUME_CKPT)]
        print(f"  ✓ Resuming from {RESUME_CKPT}", flush=True)

    rc = run(train_cmd, "Training OPT-D v2 (200 epochs)")
    if rc != 0:
        print("Training failed! Exiting.", flush=True)
        sys.exit(1)

    # --- Phase 2: Full eval on FIVES ---
    print("\n📊 PHASE 2: FULL EVAL ON FIVES", flush=True)
    rc = run([
        VENV_PYTHON, "-m", "fractal_swin_unet.eval",
        "--config", CONFIG,
        "--run_id", RUN_ID,
    ], "Full eval with threshold sweep + clDice", timeout=7200)

    # --- Phase 3: Cross-dataset eval ---
    print("\n🌍 PHASE 3: CROSS-DATASET EVAL", flush=True)

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
            "--config", CONFIG,
            "--run_id", RUN_ID,
            "--set", f"data.manifest_path={manifest}",
            "--set", f"data.dataset_root={ds_root}",
        ], f"Cross-dataset eval on {ds_name}", timeout=7200)

    # --- Phase 4: Generate predictions ---
    print("\n🖼️  PHASE 4: GENERATE PREDICTIONS", flush=True)
    run([
        VENV_PYTHON, "-m", "fractal_swin_unet.infer",
        "--config", CONFIG,
        "--run_id", RUN_ID,
        "--out_dir", f"runs/{RUN_ID}/predictions",
    ], "Generating prediction maps", timeout=7200)

    # --- Summary ---
    print("\n" + "✅" * 20, flush=True)
    metrics_path = RUN_DIR / "metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        print(f"  Best Dice:   {m.get('best_full_eval_dice', 'N/A')}")
        print(f"  Best clDice: {m.get('best_cldice', m.get('test_cldice_tau_star', 'N/A'))}")
        print(f"  τ*:          {m.get('best_full_eval_tau_star', 'N/A')}")
        print(f"  Best Epoch:  {m.get('best_full_eval_epoch', 'N/A')}")
    print("DONE!", flush=True)


if __name__ == "__main__":
    main()
