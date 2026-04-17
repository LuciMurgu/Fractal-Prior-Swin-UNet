#!/usr/bin/env python3
"""Chain: OPT-F (lacunarity) → eval → OPT-G (percolation) → eval → comparison.

Runs the lacunarity and percolation ablation experiments sequentially,
producing full journal-grade metrics for each configuration.
"""
import subprocess
import sys
import os
import json
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
VENV_PYTHON = str(PROJECT / ".venv" / "bin" / "python3")

CONFIGS = [
    ("opt_F_lacunarity", "configs/optim/OPT_F_lacunarity.yaml", "LFD + Lacunarity"),
    ("opt_G_percolation", "configs/optim/OPT_G_percolation.yaml", "LFD + Lac + Percolation"),
]


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

    for run_id, config, label in CONFIGS:
        run_dir = PROJECT / "runs" / run_id

        # --- Check if already done ---
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            m = json.loads(metrics_path.read_text())
            dice = m.get("best_full_eval_dice", m.get("best_dice", "?"))
            print(f"\n✅ {label} already completed. Dice = {dice}", flush=True)
            continue

        # --- Train ---
        print(f"\n🔬 TRAINING: {label} ({run_id})", flush=True)
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        rc = run([
            VENV_PYTHON, "-m", "fractal_swin_unet.train",
            "--config", config,
            "--run_id", run_id,
        ], f"Training {label}")
        if rc != 0:
            print(f"  ⚠ Training failed for {label}!", flush=True)
            continue

        # --- Eval ---
        print(f"\n📊 EVAL: {label} ({run_id})", flush=True)
        best_ckpt = run_dir / "checkpoint_best.pt"
        final_ckpt = run_dir / "checkpoint.pt"
        ckpt = str(best_ckpt) if best_ckpt.exists() else str(final_ckpt)

        rc = run([
            VENV_PYTHON, "-m", "fractal_swin_unet.eval",
            "--config", config,
            "--checkpoint", ckpt,
            "--run_id", f"{run_id}_eval",
        ], f"Evaluating {label}", timeout=7200)

        # Clean GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(10)

    # --- Final comparison table ---
    print("\n" + "=" * 80, flush=True)
    print("  ABLATION COMPARISON: FRACTAL PRIOR COMPONENTS", flush=True)
    print("=" * 80, flush=True)

    all_runs = [
        ("baseline_200ep", "Baseline (no prior)"),
        ("opt_D_v2_200ep", "OPT-D v2 (LFD only)"),
        ("opt_F_lacunarity", "OPT-F (LFD + Lac)"),
        ("opt_F_lacunarity_eval", "  └─ eval"),
        ("opt_G_percolation", "OPT-G (LFD+Lac+Perc)"),
        ("opt_G_percolation_eval", "  └─ eval"),
    ]

    header = f"{'Config':<25} {'Dice':>7} {'clDice':>7} {'AUC':>7} {'Se':>7} {'Sp':>7} {'Acc':>7} {'F1':>7}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for run_id, label in all_runs:
        mp = PROJECT / "runs" / run_id / "metrics.json"
        if not mp.exists():
            print(f"  {label:<25} {'—':>7} {'—':>7} {'—':>7} {'—':>7} {'—':>7} {'—':>7} {'—':>7}")
            continue
        m = json.loads(mp.read_text())

        # Try test metrics first, then val metrics
        dice = m.get("test_dice_tau_star", m.get("best_full_eval_dice", m.get("best_dice", "?")))
        cldice = m.get("test_cldice_tau_star", m.get("best_cldice", "?"))
        auc = m.get("test_auc_roc", m.get("auc_roc", "?"))
        se = m.get("test_se_tau_star", m.get("se_tau_star", "?"))
        sp = m.get("test_sp_tau_star", m.get("sp_tau_star", "?"))
        acc = m.get("test_acc_tau_star", m.get("acc_tau_star", "?"))
        f1 = m.get("test_f1_tau_star", m.get("f1_tau_star", "?"))

        def fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else str(v)

        print(f"  {label:<25} {fmt(dice):>7} {fmt(cldice):>7} {fmt(auc):>7} {fmt(se):>7} {fmt(sp):>7} {fmt(acc):>7} {fmt(f1):>7}")

    print("\nDONE! 🎉", flush=True)


if __name__ == "__main__":
    main()
