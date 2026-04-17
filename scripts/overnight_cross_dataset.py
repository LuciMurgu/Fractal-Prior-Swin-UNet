#!/usr/bin/env python3
"""Overnight cross-dataset evaluation pipeline.

Evaluates FIVES-trained models (baseline + fractal) on ALL available
datasets: FIVES, DRIVE, CHASE_DB1, STARE, HRF.

Reports Dice + clDice for each (model × dataset) combination.
Resume-aware: skips any combination that already has a results JSON.

Usage:
    source .venv/bin/activate
    python -u scripts/overnight_cross_dataset.py 2>&1 | tee overnight.log
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fractal_swin_unet.config import load_config
from fractal_swin_unet.data.registry import build_dataset
from fractal_swin_unet.device import resolve_device
from fractal_swin_unet.exp.split import assign_splits
from fractal_swin_unet.inference.tiling import tiled_predict_proba
from fractal_swin_unet.metrics import cldice_score, dice_score
from fractal_swin_unet.model_factory import build_model
from fractal_swin_unet.seed import set_deterministic_seed

# ── Configuration ──────────────────────────────────────────────────────

RESULTS_DIR = Path("reports/cross_dataset")

# Models trained on FIVES
MODELS = {
    "baseline": {
        "checkpoint": "runs/xdataset_fives_baseline/checkpoint_best.pt",
        "config": "configs/cross_dataset/fives_baseline.yaml",
    },
    "fractal": {
        "checkpoint": "runs/xdataset_fives_fractal/checkpoint_best.pt",
        "config": "configs/cross_dataset/fives_fractal.yaml",
    },
}

# Datasets to evaluate on (dataset_name → overrides for config)
EVAL_DATASETS = {
    "FIVES": {
        "manifest_path": "manifests/fives.jsonl",
        "dataset_root": "/home/lucian/medical-imaging-ai/Retina Datasets/Full retina dataset/FIVES",
        "fov_enabled": False,
        "split": {"mode": "sample", "seed": 123, "ratios": {"train": 0.8, "val": 0.1, "test": 0.1}},
        "use_all_as_test": False,  # Use only val+test splits
    },
    "DRIVE": {
        "manifest_path": "manifests/drive.jsonl",
        "dataset_root": "/home/lucian/medical-imaging-ai/Retina Datasets/Full retina dataset/DRIVE",
        "fov_enabled": True,
        "split": None,  # Use manifest splits (pre-assigned)
        "use_all_as_test": True,  #  Evaluate ALL images (foreign dataset)
    },
    "CHASE_DB1": {
        "manifest_path": "manifests/chase.jsonl",
        "dataset_root": "/home/lucian/medical-imaging-ai/Retina Datasets/Full retina dataset/CHASE_DB1",
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
        "manifest_path": "manifests/hrf.jsonl",
        "dataset_root": "/home/lucian/Retina Datasets/HRF",
        "fov_enabled": True,
        "split": {"mode": "sample", "seed": 123, "ratios": {"train": 0.6, "val": 0.2, "test": 0.2}},
        "use_all_as_test": True,  # Evaluate all HRF images (foreign dataset)
    },
}

TAU_GRID = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]


def _cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def evaluate_model_on_dataset(
    model_name: str,
    model_cfg: dict,
    dataset_name: str,
    dataset_cfg: dict,
    device: torch.device,
) -> dict:
    """Evaluate a single (model, dataset) pair. Returns metrics dict."""

    results_file = RESULTS_DIR / f"{model_name}_on_{dataset_name}.json"
    if results_file.exists():
        print(f"  ⏭  SKIP {model_name} on {dataset_name} — already done")
        with open(results_file) as f:
            return json.load(f)

    print(f"  🔬 Evaluating {model_name} on {dataset_name}...")
    t0 = time.time()

    # Load model config and override data paths
    cfg = load_config(model_cfg["config"])
    cfg["data"]["manifest_path"] = dataset_cfg["manifest_path"]
    cfg["data"]["dataset_root"] = dataset_cfg["dataset_root"]
    cfg["eval"]["fov"]["enabled"] = dataset_cfg["fov_enabled"]
    if dataset_cfg.get("split"):
        cfg["data"]["split"] = dataset_cfg["split"]

    set_deterministic_seed(42)

    # Build model + load checkpoint
    model = build_model(cfg).to(device)
    ckpt_path = model_cfg["checkpoint"]
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()
    del state

    # Build dataset
    dataset = build_dataset(cfg)

    # Assign splits if needed
    if dataset_cfg.get("split"):
        split = dataset_cfg["split"]
        dataset.manifest = assign_splits(
            dataset.manifest,
            mode=split["mode"],
            seed=split["seed"],
            ratios=split["ratios"],
        )

    # Determine which images to evaluate
    if dataset_cfg.get("use_all_as_test"):
        eval_indices = list(range(len(dataset)))
    else:
        # Use val+test only (for in-distribution FIVES)
        eval_indices = [
            i for i, s in enumerate(dataset.manifest)
            if s.get("split") in ("val", "test")
        ]
    if not eval_indices:
        eval_indices = list(range(len(dataset)))

    # Tiled inference config
    tiled_cfg = cfg.get("infer", {}).get("tiled", {})
    ph, pw = tiled_cfg.get("patch_size", [384, 384])
    sh, sw = tiled_cfg.get("stride", [192, 192])
    fractal_cfg = cfg.get("fractal_prior", {})

    # Collect predictions
    probs_list, gts_list = [], []
    per_image_metrics = []

    with torch.no_grad():
        for j, idx in enumerate(eval_indices):
            sample = dataset[idx]
            image = sample["image"]
            gt = sample["mask"]
            sample_id = sample.get("id", f"sample_{idx}")

            probs = tiled_predict_proba(
                model, image,
                patch_size=(int(ph), int(pw)),
                stride=(int(sh), int(sw)),
                blend=str(tiled_cfg.get("blend", "hann")),
                pad_mode=str(tiled_cfg.get("pad_mode", "reflect")),
                batch_tiles=int(tiled_cfg.get("batch_tiles", 2)),
                device=device,
                eps=float(tiled_cfg.get("eps", 1e-6)),
                fractal_prior_config=fractal_cfg,
            )

            probs_list.append(probs.cpu())
            gts_list.append(gt.unsqueeze(0).cpu())

            if (j + 1) % 5 == 0 or (j + 1) == len(eval_indices):
                elapsed = time.time() - t0
                eta = elapsed / (j + 1) * (len(eval_indices) - j - 1)
                print(f"    [{j+1}/{len(eval_indices)}] {sample_id} "
                      f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    all_probs = torch.cat(probs_list, dim=0)
    all_gts = torch.cat(gts_list, dim=0)

    # Threshold sweep
    best_dice, best_tau = 0.0, 0.5
    for tau in TAU_GRID:
        d = float(dice_score((all_probs >= tau).float(), all_gts).item())
        if d > best_dice:
            best_dice, best_tau = d, tau

    # Compute metrics at best tau and at 0.5
    preds_tau = (all_probs >= best_tau).float()
    preds_05 = (all_probs >= 0.5).float()

    # Per-image metrics
    for j in range(all_probs.shape[0]):
        p = all_probs[j:j+1]
        g = all_gts[j:j+1]
        pred_t = (p >= best_tau).float()
        pred_5 = (p >= 0.5).float()
        per_image_metrics.append({
            "index": j,
            "dice_tau_star": float(dice_score(pred_t, g).item()),
            "dice_at_0_5": float(dice_score(pred_5, g).item()),
            "cldice_tau_star": float(cldice_score(pred_t, g).item()),
            "cldice_at_0_5": float(cldice_score(pred_5, g).item()),
        })

    results = {
        "model": model_name,
        "dataset": dataset_name,
        "n_images": len(eval_indices),
        "tau_star": best_tau,
        "dice_tau_star": best_dice,
        "dice_at_0_5": float(dice_score(preds_05, all_gts).item()),
        "cldice_tau_star": float(cldice_score(preds_tau, all_gts).item()),
        "cldice_at_0_5": float(cldice_score(preds_05, all_gts).item()),
        "per_image": per_image_metrics,
        "elapsed_seconds": time.time() - t0,
    }

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  ✅ {model_name} on {dataset_name}: "
          f"Dice={best_dice:.4f} clDice={results['cldice_tau_star']:.4f} "
          f"τ*={best_tau:.2f} ({len(eval_indices)} images, {results['elapsed_seconds']:.0f}s)")

    # Cleanup
    del model, all_probs, all_gts, probs_list, gts_list
    _cleanup_gpu()

    return results


def generate_summary(all_results: list[dict]):
    """Generate CSV + markdown summary tables."""

    # CSV
    csv_path = RESULTS_DIR / "cross_dataset_summary.csv"
    with open(csv_path, "w") as f:
        f.write("model,dataset,n_images,tau_star,dice_tau_star,dice_at_0_5,cldice_tau_star,cldice_at_0_5\n")
        for r in all_results:
            f.write(f"{r['model']},{r['dataset']},{r['n_images']},"
                    f"{r['tau_star']:.2f},{r['dice_tau_star']:.4f},"
                    f"{r['dice_at_0_5']:.4f},{r['cldice_tau_star']:.4f},"
                    f"{r['cldice_at_0_5']:.4f}\n")

    # Markdown
    md_path = RESULTS_DIR / "cross_dataset_summary.md"
    with open(md_path, "w") as f:
        f.write("# Cross-Dataset Generalization Results\n\n")
        f.write("Train: FIVES (800 images) | Eval: tiled full-image with threshold sweep\n\n")
        f.write("| Model | Dataset | n | τ* | **Dice** | Dice@0.5 | **clDice** | clDice@0.5 |\n")
        f.write("|-------|---------|---|----|----------|----------|------------|------------|\n")
        for r in all_results:
            f.write(f"| {r['model']} | {r['dataset']} | {r['n_images']} | "
                    f"{r['tau_star']:.2f} | **{r['dice_tau_star']:.4f}** | "
                    f"{r['dice_at_0_5']:.4f} | **{r['cldice_tau_star']:.4f}** | "
                    f"{r['cldice_at_0_5']:.4f} |\n")

        # Compute deltas
        f.write("\n## Fractal vs Baseline Delta\n\n")
        f.write("| Dataset | ΔDice | ΔclDice | Verdict |\n")
        f.write("|---------|-------|---------|--------|\n")
        baseline_by_ds = {r['dataset']: r for r in all_results if r['model'] == 'baseline'}
        fractal_by_ds = {r['dataset']: r for r in all_results if r['model'] == 'fractal'}
        for ds in EVAL_DATASETS:
            if ds in baseline_by_ds and ds in fractal_by_ds:
                b = baseline_by_ds[ds]
                fr = fractal_by_ds[ds]
                dd = fr['dice_tau_star'] - b['dice_tau_star']
                dc = fr['cldice_tau_star'] - b['cldice_tau_star']
                verdict = "✅ Fractal wins" if dd > 0 else "❌ Baseline wins"
                f.write(f"| {ds} | {dd:+.4f} | {dc:+.4f} | {verdict} |\n")

    print(f"\n📊 Summary saved to:\n  {csv_path}\n  {md_path}")


def main():
    print("=" * 70)
    print("  OVERNIGHT CROSS-DATASET EVALUATION PIPELINE")
    print("  Train: FIVES (800 images)")
    print("  Eval: FIVES, DRIVE, CHASE_DB1, STARE, HRF")
    print("  Models: baseline (no fractal) vs fractal (SPADE v2 + PDE)")
    print("=" * 70)

    device = resolve_device()
    print(f"Device: {device}")
    print(f"Results dir: {RESULTS_DIR}")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    total_combos = len(MODELS) * len(EVAL_DATASETS)
    combo_idx = 0

    for model_name, model_cfg in MODELS.items():
        ckpt_path = Path(model_cfg["checkpoint"])
        if not ckpt_path.exists():
            print(f"⚠️  SKIP model '{model_name}': checkpoint not found at {ckpt_path}")
            continue

        for dataset_name, dataset_cfg in EVAL_DATASETS.items():
            combo_idx += 1
            print(f"\n{'─'*70}")
            print(f"  [{combo_idx}/{total_combos}] {model_name} → {dataset_name}")
            print(f"{'─'*70}")

            try:
                result = evaluate_model_on_dataset(
                    model_name, model_cfg,
                    dataset_name, dataset_cfg,
                    device,
                )
                all_results.append(result)
            except Exception as e:
                print(f"  ❌ FAILED: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "model": model_name,
                    "dataset": dataset_name,
                    "error": str(e),
                    "n_images": 0,
                    "tau_star": 0,
                    "dice_tau_star": 0,
                    "dice_at_0_5": 0,
                    "cldice_tau_star": 0,
                    "cldice_at_0_5": 0,
                })

            _cleanup_gpu()

    # Generate summary tables
    generate_summary(all_results)

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
