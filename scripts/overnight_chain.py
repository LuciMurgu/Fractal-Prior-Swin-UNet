#!/usr/bin/env python3
"""Crash-safe overnight chain: clDice audit → cross-dataset generalization.

Restart-safe design:
  - Every per-image result saved to JSONL immediately (no data lost on crash)
  - Combo-level results saved as JSON (resume skips completed combos)
  - GPU cleanup between every model load
  - Designed to be wrapped by restart_wrapper.sh

Usage:
    source .venv/bin/activate
    python -u scripts/overnight_chain.py 2>&1 | tee -a overnight_chain.log

Or with crash-safe wrapper:
    bash scripts/restart_wrapper.sh
"""
from __future__ import annotations

import gc
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

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

# ── Results directory ──────────────────────────────────────────────────
RESULTS_DIR = Path("reports/overnight_chain")
PARTIAL_DIR = RESULTS_DIR / "partial"   # Per-image incremental saves

TAU_GRID = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

# ── Phase 1: clDice audit of all 8 ablation configs on HRF ────────────
ABLATION_CONFIGS = {
    "A_baseline":    "configs/ablation_v2/A_baseline.yaml",
    "B_lfd_gate":    "configs/ablation_v2/B_lfd_gate.yaml",
    "C_lfd_spade1":  "configs/ablation_v2/C_lfd_spade1.yaml",
    "D_pde_spade2":  "configs/ablation_v2/D_pde_spade2.yaml",
    "E_hessian":     "configs/ablation_v2/E_hessian.yaml",
    "F_skel_recall": "configs/ablation_v2/F_skel_recall.yaml",
    "G_fractal_bce": "configs/ablation_v2/G_fractal_bce.yaml",
    "H_full":        "configs/ablation_v2/H_full.yaml",
}

ABLATION_CHECKPOINTS = {
    name: f"runs/ablation_v2_{name}/checkpoint_best.pt"
    for name in ABLATION_CONFIGS
}

# ── Phase 2: Cross-dataset generalization ──────────────────────────────
# Models are selected dynamically from Phase 1 winner + baseline
TRAIN_CONFIGS = {
    "baseline": {
        "config": "configs/cross_dataset/fives_baseline.yaml",
        "checkpoint": "runs/xdataset_fives_baseline/checkpoint_best.pt",
    },
    "fractal": {
        "config": "configs/cross_dataset/fives_fractal.yaml",
        "checkpoint": "runs/xdataset_fives_fractal/checkpoint_best.pt",
    },
}

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


def _cleanup_gpu():
    """Aggressive GPU memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _combo_key(model_name: str, dataset_name: str) -> str:
    return f"{model_name}_on_{dataset_name}"


def _result_file(phase: str, combo_key: str) -> Path:
    return RESULTS_DIR / phase / f"{combo_key}.json"


def _partial_file(phase: str, combo_key: str) -> Path:
    return PARTIAL_DIR / phase / f"{combo_key}.jsonl"


def _is_combo_done(phase: str, combo_key: str) -> bool:
    return _result_file(phase, combo_key).exists()


def _count_partial(phase: str, combo_key: str) -> int:
    """Count how many images already have partial results."""
    pf = _partial_file(phase, combo_key)
    if not pf.exists():
        return 0
    with open(pf) as f:
        return sum(1 for _ in f)


def _load_partial(phase: str, combo_key: str) -> list[dict]:
    """Load all partial per-image results."""
    pf = _partial_file(phase, combo_key)
    if not pf.exists():
        return []
    results = []
    with open(pf) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def _save_partial_image(phase: str, combo_key: str, image_result: dict):
    """Append one image result to the partial JSONL — crash-safe."""
    pf = _partial_file(phase, combo_key)
    pf.parent.mkdir(parents=True, exist_ok=True)
    with open(pf, "a") as f:
        f.write(json.dumps(image_result) + "\n")


def _finalize_combo(
    phase: str,
    combo_key: str,
    model_name: str,
    dataset_name: str,
    partial_results: list[dict],
    elapsed: float,
):
    """Aggregate per-image partials into final combo result JSON."""
    if not partial_results:
        return {}

    # Collect all probs and gts from saved per-image data
    # We already have per-image metrics — just aggregate
    n = len(partial_results)

    # Threshold sweep on aggregated per-image dice
    best_dice, best_tau = 0.0, 0.5
    for tau in TAU_GRID:
        tau_key = f"dice_tau_{tau:.2f}"
        if tau_key in partial_results[0]:
            d = sum(r[tau_key] for r in partial_results) / n
            if d > best_dice:
                best_dice, best_tau = d, tau

    # Aggregate at best tau and at 0.5
    dice_05 = sum(r.get("dice_tau_0.50", 0) for r in partial_results) / n
    cldice_best = sum(r.get(f"cldice_tau_{best_tau:.2f}", 0) for r in partial_results) / n
    cldice_05 = sum(r.get("cldice_tau_0.50", 0) for r in partial_results) / n

    result = {
        "model": model_name,
        "dataset": dataset_name,
        "n_images": n,
        "tau_star": best_tau,
        "dice_tau_star": best_dice,
        "dice_at_0_5": dice_05,
        "cldice_tau_star": cldice_best,
        "cldice_at_0_5": cldice_05,
        "elapsed_seconds": elapsed,
        "per_image": partial_results,
    }

    rf = _result_file(phase, combo_key)
    rf.parent.mkdir(parents=True, exist_ok=True)
    with open(rf, "w") as f:
        json.dump(result, f, indent=2)

    return result


def evaluate_combo(
    phase: str,
    model_name: str,
    config_path: str,
    checkpoint_path: str,
    dataset_name: str,
    dataset_cfg: dict,
    device: torch.device,
) -> dict:
    """Evaluate one (model, dataset) combo with per-image crash safety."""

    combo_key = _combo_key(model_name, dataset_name)

    # Skip if fully done
    if _is_combo_done(phase, combo_key):
        print(f"  ⏭  SKIP {combo_key} — already done")
        with open(_result_file(phase, combo_key)) as f:
            return json.load(f)

    print(f"  🔬 Evaluating {combo_key}...")
    t0 = time.time()

    # Check for partial progress
    n_done = _count_partial(phase, combo_key)
    if n_done > 0:
        print(f"  ↩  Resuming from image {n_done} (found partial results)")

    # Load config with dataset overrides
    cfg = load_config(config_path)

    # Override data paths for cross-dataset eval
    if dataset_cfg:
        cfg["data"]["manifest_path"] = dataset_cfg["manifest_path"]
        cfg["data"]["dataset_root"] = dataset_cfg["dataset_root"]
        cfg["eval"]["fov"]["enabled"] = dataset_cfg["fov_enabled"]
        if dataset_cfg.get("split"):
            cfg["data"]["split"] = dataset_cfg["split"]

    set_deterministic_seed(42)

    # Build model + load checkpoint
    model = build_model(cfg).to(device)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()
    del state
    _cleanup_gpu()

    # Build dataset
    dataset = build_dataset(cfg)

    # Assign splits if needed
    if dataset_cfg and dataset_cfg.get("split"):
        split = dataset_cfg["split"]
        dataset.manifest = assign_splits(
            dataset.manifest,
            mode=split["mode"],
            seed=split["seed"],
            ratios=split["ratios"],
        )

    # Determine eval indices
    if dataset_cfg and dataset_cfg.get("use_all_as_test"):
        eval_indices = list(range(len(dataset)))
    elif dataset_cfg:
        eval_indices = [
            i for i, s in enumerate(dataset.manifest)
            if s.get("split") in ("val", "test")
        ]
        if not eval_indices:
            eval_indices = list(range(len(dataset)))
    else:
        # Ablation eval: use val+test from the config's own split
        split_cfg = cfg.get("data", {}).get("split", {})
        dataset.manifest = assign_splits(
            dataset.manifest,
            mode=split_cfg.get("mode", "sample"),
            seed=int(split_cfg.get("seed", 42)),
            ratios=split_cfg.get("ratios", {"train": 0.6, "val": 0.2, "test": 0.2}),
        )
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

    # FOV mask handling
    fov_enabled = cfg.get("eval", {}).get("fov", {}).get("enabled", False)

    with torch.no_grad():
        for j, idx in enumerate(eval_indices):
            # Skip already-processed images
            if j < n_done:
                continue

            sample = dataset[idx]
            image = sample["image"]
            gt = sample["mask"]
            sample_id = sample.get("id", f"sample_{idx}")
            fov_mask = sample.get("fov_mask", None) if fov_enabled else None

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

            # Per-image metrics at ALL thresholds (cheap: just comparisons)
            probs_cpu = probs.cpu()
            gt_cpu = gt.unsqueeze(0).cpu()
            fov_t = None
            if fov_mask is not None:
                fov_t = fov_mask.unsqueeze(0).unsqueeze(0).cpu() if fov_mask.ndim == 2 else fov_mask.unsqueeze(0).cpu()

            image_result = {"index": j, "dataset_idx": idx, "id": sample_id}
            for tau in TAU_GRID:
                pred = (probs_cpu >= tau).float()
                d = float(dice_score(pred, gt_cpu, mask=fov_t).item())
                c = float(cldice_score(pred, gt_cpu, mask=fov_t).item())
                image_result[f"dice_tau_{tau:.2f}"] = d
                image_result[f"cldice_tau_{tau:.2f}"] = c

            # Save immediately — crash-safe
            _save_partial_image(phase, combo_key, image_result)

            if (j + 1) % 3 == 0 or (j + 1) == len(eval_indices):
                elapsed = time.time() - t0
                rate = elapsed / (j - n_done + 1) if j >= n_done else 0
                eta = rate * (len(eval_indices) - j - 1)
                print(f"    [{j+1}/{len(eval_indices)}] {sample_id} "
                      f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    # Finalize: aggregate all partials into one JSON
    all_partials = _load_partial(phase, combo_key)
    result = _finalize_combo(
        phase, combo_key, model_name, dataset_name,
        all_partials, time.time() - t0,
    )

    dice_star = result.get("dice_tau_star", 0)
    cldice_star = result.get("cldice_tau_star", 0)
    tau_star = result.get("tau_star", 0.5)
    print(f"  ✅ {combo_key}: "
          f"Dice={dice_star:.4f} clDice={cldice_star:.4f} "
          f"τ*={tau_star:.2f} ({len(all_partials)} images)")

    # Cleanup
    del model
    _cleanup_gpu()

    return result


def generate_summary(phase: str, all_results: list[dict]):
    """Generate CSV + markdown summary."""
    out_dir = RESULTS_DIR / phase
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w") as f:
        f.write("model,dataset,n_images,tau_star,dice_tau_star,dice_at_0_5,cldice_tau_star,cldice_at_0_5\n")
        for r in all_results:
            if "error" in r:
                continue
            f.write(f"{r['model']},{r['dataset']},{r['n_images']},"
                    f"{r['tau_star']:.2f},{r['dice_tau_star']:.4f},"
                    f"{r['dice_at_0_5']:.4f},{r['cldice_tau_star']:.4f},"
                    f"{r['cldice_at_0_5']:.4f}\n")

    # Markdown
    md_path = out_dir / "summary.md"
    with open(md_path, "w") as f:
        f.write(f"# {phase.replace('_', ' ').title()} Results\n\n")
        f.write("| Model | Dataset | n | τ* | **Dice** | Dice@0.5 | **clDice** | clDice@0.5 |\n")
        f.write("|-------|---------|---|----|----------|----------|------------|------------|\n")
        for r in all_results:
            if "error" in r:
                f.write(f"| {r['model']} | {r['dataset']} | ERROR | | | | | |\n")
                continue
            f.write(f"| {r['model']} | {r['dataset']} | {r['n_images']} | "
                    f"{r['tau_star']:.2f} | **{r['dice_tau_star']:.4f}** | "
                    f"{r['dice_at_0_5']:.4f} | **{r['cldice_tau_star']:.4f}** | "
                    f"{r['cldice_at_0_5']:.4f} |\n")

    print(f"\n📊 Summary saved: {csv_path}, {md_path}")


# ══════════════════════════════════════════════════════════════════════
# PHASE 1: clDice audit — all 8 ablation configs on HRF
# ══════════════════════════════════════════════════════════════════════

def run_phase1(device: torch.device) -> list[dict]:
    """Evaluate clDice for all 8 ablation checkpoints on HRF val+test."""
    print("\n" + "=" * 70)
    print("  PHASE 1: clDice AUDIT — 8 ablation configs on HRF")
    print("  Goal: find the clDice winner (not just Dice winner)")
    print("=" * 70)

    phase = "phase1_cldice_audit"
    all_results = []

    for i, (name, config_path) in enumerate(ABLATION_CONFIGS.items()):
        ckpt = ABLATION_CHECKPOINTS[name]
        if not Path(ckpt).exists():
            print(f"\n⚠️  SKIP {name}: checkpoint not found at {ckpt}")
            continue

        print(f"\n{'─'*70}")
        print(f"  [{i+1}/{len(ABLATION_CONFIGS)}] {name}")
        print(f"{'─'*70}")

        try:
            result = evaluate_combo(
                phase=phase,
                model_name=name,
                config_path=config_path,
                checkpoint_path=ckpt,
                dataset_name="HRF",
                dataset_cfg=None,  # Use the config's own data settings
                device=device,
            )
            all_results.append(result)
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            import traceback; traceback.print_exc()
            all_results.append({"model": name, "dataset": "HRF", "error": str(e),
                                "n_images": 0, "tau_star": 0, "dice_tau_star": 0,
                                "dice_at_0_5": 0, "cldice_tau_star": 0, "cldice_at_0_5": 0})

        _cleanup_gpu()

    generate_summary(phase, all_results)

    # Print ranking
    valid = [r for r in all_results if "error" not in r]
    if valid:
        by_cldice = sorted(valid, key=lambda r: r["cldice_tau_star"], reverse=True)
        print("\n🏆 clDice RANKING (HRF val+test):")
        for i, r in enumerate(by_cldice):
            marker = " ← WINNER" if i == 0 else ""
            print(f"  {i+1}. {r['model']}: clDice={r['cldice_tau_star']:.4f} "
                  f"Dice={r['dice_tau_star']:.4f} τ*={r['tau_star']:.2f}{marker}")

    return all_results


# ══════════════════════════════════════════════════════════════════════
# PHASE 2: Cross-dataset generalization (FIVES-trained → all datasets)
# ══════════════════════════════════════════════════════════════════════

def run_phase2(device: torch.device) -> list[dict]:
    """Cross-dataset eval: FIVES-trained baseline+fractal on all 5 datasets."""
    print("\n" + "=" * 70)
    print("  PHASE 2: CROSS-DATASET GENERALIZATION")
    print("  Train: FIVES | Eval: FIVES, DRIVE, CHASE, STARE, HRF")
    print("  Models: baseline vs fractal (SPADE v2 + PDE)")
    print("=" * 70)

    phase = "phase2_cross_dataset"
    all_results = []

    for model_name, model_cfg in TRAIN_CONFIGS.items():
        ckpt = model_cfg["checkpoint"]
        if not Path(ckpt).exists():
            print(f"\n⚠️  SKIP model '{model_name}': checkpoint not found at {ckpt}")
            continue

        for ds_name, ds_cfg in EVAL_DATASETS.items():
            combo_key = _combo_key(model_name, ds_name)
            print(f"\n{'─'*70}")
            print(f"  {model_name} → {ds_name}")
            print(f"{'─'*70}")

            try:
                result = evaluate_combo(
                    phase=phase,
                    model_name=model_name,
                    config_path=model_cfg["config"],
                    checkpoint_path=ckpt,
                    dataset_name=ds_name,
                    dataset_cfg=ds_cfg,
                    device=device,
                )
                all_results.append(result)
            except Exception as e:
                print(f"  ❌ FAILED: {e}")
                import traceback; traceback.print_exc()
                all_results.append({"model": model_name, "dataset": ds_name, "error": str(e),
                                    "n_images": 0, "tau_star": 0, "dice_tau_star": 0,
                                    "dice_at_0_5": 0, "cldice_tau_star": 0, "cldice_at_0_5": 0})

            _cleanup_gpu()

    generate_summary(phase, all_results)

    # Print delta table
    valid = [r for r in all_results if "error" not in r]
    baseline_by_ds = {r["dataset"]: r for r in valid if r["model"] == "baseline"}
    fractal_by_ds = {r["dataset"]: r for r in valid if r["model"] == "fractal"}

    if baseline_by_ds and fractal_by_ds:
        print("\n🏆 Fractal vs Baseline Δ:")
        print(f"  {'Dataset':<12} {'ΔDice':>8} {'ΔclDice':>8} {'Verdict'}")
        print(f"  {'─'*48}")
        for ds in EVAL_DATASETS:
            if ds in baseline_by_ds and ds in fractal_by_ds:
                b = baseline_by_ds[ds]
                fr = fractal_by_ds[ds]
                dd = fr["dice_tau_star"] - b["dice_tau_star"]
                dc = fr["cldice_tau_star"] - b["cldice_tau_star"]
                v = "✅ Fractal" if dc > 0 else "❌ Baseline"
                print(f"  {ds:<12} {dd:>+8.4f} {dc:>+8.4f} {v}")

    return all_results


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  OVERNIGHT CHAIN — CRASH-SAFE")
    print("  Phase 1: clDice audit (8 ablation configs × HRF)")
    print("  Phase 2: Cross-dataset generalization (2 models × 5 datasets)")
    print("  Per-image saves — safe to restart at any point")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)

    device = resolve_device()
    print(f"Device: {device}")
    print(f"Results: {RESULTS_DIR}")
    print()

    # Phase 1: clDice audit
    phase1_results = run_phase1(device)

    # Phase 2: Cross-dataset generalization
    phase2_results = run_phase2(device)

    print("\n" + "=" * 70)
    print("  OVERNIGHT CHAIN COMPLETE")
    print(f"  Phase 1: {len([r for r in phase1_results if 'error' not in r])}/8 configs evaluated")
    print(f"  Phase 2: {len([r for r in phase2_results if 'error' not in r])}/10 combos evaluated")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
