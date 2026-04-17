#!/usr/bin/env python3
"""Overnight: Train Config F on FIVES → Evaluate cross-dataset.

Phase 1: Train Config F on FIVES (80 epochs, ~4-5h)
Phase 2: Evaluate Config F on all 5 datasets (reuses overnight_chain logic)
Phase 3: Re-run statistical tests comparing Config F vs baseline

Usage:
    nohup .venv/bin/python -u scripts/overnight_configF.py 2>&1 | tee -a overnight_configF.log &
"""
from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fractal_swin_unet.device import resolve_device

CONFIG_F = "configs/cross_dataset/fives_configF.yaml"
RUN_DIR = "runs/xdataset_fives_configF"
CHECKPOINT = f"{RUN_DIR}/checkpoint_best.pt"

EVAL_DATASETS = {
    "FIVES": {
        "manifest_path": "manifests/fives.jsonl",
        "dataset_root": "/home/lucian/medical-imaging-ai/Retina Datasets/Full retina dataset/FIVES",
        "fov_enabled": False,
        "split": {"mode": "sample", "seed": 123, "ratios": {"train": 0.8, "val": 0.1, "test": 0.1}},
        "use_all_as_test": False,
    },
    "DRIVE": {
        "manifest_path": "manifests/drive.jsonl",
        "dataset_root": "/home/lucian/medical-imaging-ai/Retina Datasets/Full retina dataset/DRIVE",
        "fov_enabled": True,
        "split": None,
        "use_all_as_test": True,
    },
    "CHASE_DB1": {
        "manifest_path": "manifests/chase.jsonl",
        "dataset_root": "/home/lucian/medical-imaging-ai/Retina Datasets/Full retina dataset/CHASE",
        "fov_enabled": False,
        "split": None,
        "use_all_as_test": True,
    },
    "STARE": {
        "manifest_path": "manifests/stare.jsonl",
        "dataset_root": "/home/lucian/medical-imaging-ai/Retina Datasets/Full retina dataset/STARE",
        "fov_enabled": False,
        "split": None,
        "use_all_as_test": True,
    },
    "HRF": {
        "manifest_path": "manifests/hrf_full.jsonl",
        "dataset_root": "/home/lucian/medical-imaging-ai/Retina Datasets/Full retina dataset/HRF",
        "fov_enabled": True,
        "split": {"mode": "sample", "seed": 123, "ratios": {"train": 0.6, "val": 0.2, "test": 0.2}},
        "use_all_as_test": True,
    },
}

RESULTS_DIR = Path("reports/configF_chain")
PARTIAL_DIR = RESULTS_DIR / "partial"

TAU_GRID = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]


def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def phase1_train():
    """Train Config F on FIVES."""
    print("=" * 70)
    print("  PHASE 1: TRAIN Config F on FIVES")
    print(f"  Config: {CONFIG_F}")
    print(f"  Output: {RUN_DIR}")
    print("=" * 70)

    # Check if already trained
    if Path(CHECKPOINT).exists():
        print(f"  ⏭  SKIP training — checkpoint already exists: {CHECKPOINT}")
        return True

    # Launch training
    cmd = [
        sys.executable, "-m", "fractal_swin_unet.train",
        "--config", CONFIG_F,
        "--run_id", "xdataset_fives_configF",
        "--run_base", "runs",
    ]
    print(f"  Running: {' '.join(cmd)}")
    t0 = time.time()

    result = subprocess.run(cmd, capture_output=False)

    elapsed = time.time() - t0
    print(f"\n  Training completed in {elapsed/3600:.1f}h (exit code: {result.returncode})")

    if not Path(CHECKPOINT).exists():
        print(f"  ❌ FAILED: checkpoint not found at {CHECKPOINT}")
        return False

    print(f"  ✅ Checkpoint saved: {CHECKPOINT}")
    return True


def phase2_eval():
    """Evaluate Config F on all datasets (reuses overnight_chain evaluate_combo)."""
    print("\n" + "=" * 70)
    print("  PHASE 2: EVALUATE Config F on ALL DATASETS")
    print("=" * 70)

    # Import the evaluate_combo from overnight_chain
    from overnight_chain import evaluate_combo, generate_summary, RESULTS_DIR as OC_RESULTS_DIR

    device = resolve_device()
    phase = "configF_cross_dataset"
    all_results = []

    for ds_name, ds_cfg in EVAL_DATASETS.items():
        combo_key = f"configF_on_{ds_name}"
        print(f"\n{'─'*70}")
        print(f"  configF → {ds_name}")
        print(f"{'─'*70}")

        try:
            result = evaluate_combo(
                phase=phase,
                model_name="configF",
                config_path=CONFIG_F,
                checkpoint_path=CHECKPOINT,
                dataset_name=ds_name,
                dataset_cfg=ds_cfg,
                device=device,
            )
            all_results.append(result)
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            import traceback; traceback.print_exc()
            all_results.append({
                "model": "configF", "dataset": ds_name, "error": str(e),
                "n_images": 0, "tau_star": 0, "dice_tau_star": 0,
                "dice_at_0_5": 0, "cldice_tau_star": 0, "cldice_at_0_5": 0
            })

        cleanup_gpu()

    # Save results using overnight_chain's generate_summary
    generate_summary(phase, all_results)

    # Also compare with baseline
    print("\n🏆 Config F vs Baseline Δ:")
    baseline_dir = Path("reports/overnight_chain/phase2_cross_dataset")
    print(f"  {'Dataset':<12} {'B_clDice':>10} {'F_clDice':>10} {'ΔclDice':>8} {'Verdict'}")
    print(f"  {'─'*52}")
    for ds in EVAL_DATASETS:
        b_file = baseline_dir / f"baseline_on_{ds}.json"
        f_result = next((r for r in all_results if r.get("dataset") == ds and "error" not in r), None)

        if b_file.exists() and f_result:
            b = json.load(open(b_file))
            b_cl = b["cldice_tau_star"]
            f_cl = f_result["cldice_tau_star"]
            delta = f_cl - b_cl
            v = "✅ ConfigF" if delta > 0 else "❌ Baseline"
            print(f"  {ds:<12} {b_cl:>10.4f} {f_cl:>10.4f} {delta:>+8.4f} {v}")
        else:
            print(f"  {ds:<12} {'---':>10} {'---':>10} {'---':>8}")

    return all_results


def phase3_stats():
    """Run statistical tests for Config F vs baseline."""
    print("\n" + "=" * 70)
    print("  PHASE 3: STATISTICAL TESTS (Config F vs Baseline)")
    print("=" * 70)

    cmd = [sys.executable, "scripts/statistical_analysis.py"]
    # We need a modified version that compares configF vs baseline
    # For now, just note that results are available for manual comparison
    print("  → Per-image data available in reports/overnight_chain/configF_cross_dataset/")
    print("  → Run scripts/statistical_analysis.py with modified model names to compare")


def main():
    print("=" * 70)
    print("  OVERNIGHT CONFIG F CHAIN")
    print("  Phase 1: Train Config F on FIVES (~5h)")
    print("  Phase 2: Cross-dataset evaluation (5 datasets, ~2h)")
    print("  Total estimated: ~7h")
    print("=" * 70)

    # Phase 1: Training
    ok = phase1_train()
    if not ok:
        print("\n❌ TRAINING FAILED — aborting chain")
        return 1

    cleanup_gpu()

    # Phase 2: Cross-dataset eval
    results = phase2_eval()

    # Phase 3: Stats
    phase3_stats()

    valid = [r for r in results if "error" not in r]
    print(f"\n{'='*70}")
    print(f"  OVERNIGHT CONFIG F CHAIN COMPLETE")
    print(f"  {len(valid)}/5 datasets evaluated successfully")
    print(f"{'='*70}")

    return 0


if __name__ == "__main__":
    # Add scripts/ to path for overnight_chain imports
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
