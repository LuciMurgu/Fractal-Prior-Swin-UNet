#!/usr/bin/env python3
"""Ablation evaluation script with TTA, post-processing, and comprehensive metrics.

Usage:
    python scripts/run_ablation_eval.py --ablation-dir runs/ --output results/ablation_table.csv

Evaluates all ablation checkpoints with:
- TTA (D4 group: 8 augmentations)
- Post-processing (morphological closing + small component removal)
- Threshold sweep on val set → optimal τ*
- Metrics: Dice, AUC, Sensitivity, Specificity, clDice
- FOV masking
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from fractal_swin_unet.data.registry import build_dataset
from fractal_swin_unet.eval import _build_model, _load_config
from fractal_swin_unet.exp.split import assign_splits
from fractal_swin_unet.exp.manifest import load_manifest
from fractal_swin_unet.inference.tiling import tiled_predict_proba
from fractal_swin_unet.inference.tta import tta_predict_proba
from fractal_swin_unet.inference.postprocess import postprocess_vessel_mask
from fractal_swin_unet.metrics import dice_score
from fractal_swin_unet.device import resolve_device


# ─── Metrics ──────────────────────────────────────────────────

def soft_skeletonize(mask: torch.Tensor, n_iter: int = 10) -> torch.Tensor:
    """Soft morphological skeletonization via iterative erosion.

    Args:
        mask: Binary mask (B, 1, H, W), float in {0, 1}.
        n_iter: Number of erosion iterations.

    Returns:
        Soft skeleton (B, 1, H, W), float.
    """
    kernel = torch.ones(1, 1, 3, 3, device=mask.device, dtype=mask.dtype) / 9.0
    skeleton = torch.zeros_like(mask)
    current = mask.clone()

    for _ in range(n_iter):
        eroded = F.conv2d(current, kernel, padding=1)
        eroded = (eroded > 0.5).float()
        opened = F.conv2d(eroded, kernel, padding=1)
        opened = (opened > 0.5).float()
        skeleton = torch.maximum(skeleton, current - opened)
        current = eroded
        if current.sum() < 1:
            break

    return skeleton


def compute_cldice(
    pred: torch.Tensor,
    gt: torch.Tensor,
    n_iter: int = 10,
) -> float:
    """Compute centerline Dice (clDice) between binary masks.

    clDice = 2 * Tprec * Tsens / (Tprec + Tsens)
    where Tprec = |skel(pred) ∩ gt| / |skel(pred)|
          Tsens = |skel(gt) ∩ pred| / |skel(gt)|
    """
    skel_pred = soft_skeletonize(pred, n_iter)
    skel_gt = soft_skeletonize(gt, n_iter)

    tprec = (skel_pred * gt).sum() / (skel_pred.sum() + 1e-8)
    tsens = (skel_gt * pred).sum() / (skel_gt.sum() + 1e-8)

    cldice = 2.0 * tprec * tsens / (tprec + tsens + 1e-8)
    return float(cldice.item())


def compute_auc(probs: torch.Tensor, gt: torch.Tensor) -> float:
    """Compute AUC-ROC via trapezoidal integration over 100 thresholds."""
    thresholds = torch.linspace(0, 1, 100, device=probs.device)
    tprs = []
    fprs = []
    gt_flat = gt.flatten().bool()
    probs_flat = probs.flatten()

    for t in thresholds:
        pred = probs_flat >= t
        tp = (pred & gt_flat).sum().float()
        fp = (pred & ~gt_flat).sum().float()
        fn = (~pred & gt_flat).sum().float()
        tn = (~pred & ~gt_flat).sum().float()

        tpr = tp / (tp + fn + 1e-8)
        fpr = fp / (fp + tn + 1e-8)
        tprs.append(float(tpr))
        fprs.append(float(fpr))

    # Trapezoidal integration
    auc = 0.0
    for i in range(1, len(fprs)):
        auc += abs(fprs[i] - fprs[i - 1]) * (tprs[i] + tprs[i - 1]) / 2
    return auc


def compute_all_metrics(
    pred_binary: torch.Tensor,
    probs: torch.Tensor,
    gt: torch.Tensor,
    fov: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute all evaluation metrics.

    Args:
        pred_binary: Thresholded prediction (B, 1, H, W).
        probs: Raw probabilities (B, 1, H, W).
        gt: Ground truth binary (B, 1, H, W).
        fov: FOV mask (B, 1, H, W) or None.

    Returns:
        Dict with Dice, AUC, Sensitivity, Specificity, clDice.
    """
    if fov is not None:
        fov = fov.float()
        pred_binary = pred_binary * fov
        probs = probs * fov
        gt = gt * fov

    # Dice
    dice = float(dice_score(pred_binary, gt).item())

    # AUC
    auc = compute_auc(probs, gt)

    # Sensitivity / Specificity
    pred_flat = pred_binary.flatten().bool()
    gt_flat = gt.flatten().bool()
    tp = (pred_flat & gt_flat).sum().float()
    fp = (pred_flat & ~gt_flat).sum().float()
    fn = (~pred_flat & gt_flat).sum().float()
    tn = (~pred_flat & ~gt_flat).sum().float()

    sensitivity = float(tp / (tp + fn + 1e-8))
    specificity = float(tn / (tn + fp + 1e-8))

    # clDice
    cldice = compute_cldice(pred_binary, gt, n_iter=10)

    return {
        "Dice": dice,
        "AUC": auc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "clDice": cldice,
    }


# ─── Evaluation Pipeline ─────────────────────────────────────

def find_ablation_runs(base_dir: Path) -> list[dict]:
    """Find all ablation runs with their configs and checkpoints."""
    runs = []
    # Look for configs in configs/ablation/
    config_dir = project_root / "configs" / "ablation"
    if not config_dir.exists():
        print(f"No ablation config directory found at {config_dir}")
        return runs

    for config_path in sorted(config_dir.glob("*.yaml")):
        name = config_path.stem
        # Find matching run directory
        matching_dirs = list(base_dir.glob(f"*{name}*"))
        checkpoint = None
        for d in matching_dirs:
            best = d / "checkpoint_best.pt"
            if best.exists():
                checkpoint = best
                break
        runs.append({
            "name": name,
            "config_path": str(config_path),
            "checkpoint": str(checkpoint) if checkpoint else None,
            "status": "ready" if checkpoint else "no_checkpoint",
        })
    return runs


def evaluate_config(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
    use_tta: bool = True,
    use_postprocess: bool = True,
    min_component_size: int = 200,
    close_radius: int = 3,
) -> dict[str, float]:
    """Run full evaluation pipeline for one ablation config."""
    cfg = _load_config(config_path)
    eval_cfg = cfg.get("eval", {})
    infer_cfg = cfg.get("infer", {}).get("tiled", {})

    # Load model
    model = _build_model(cfg).to(device)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()

    # Load dataset
    dataset = build_dataset(cfg)
    data_cfg = cfg.get("data", {})
    split_cfg = data_cfg.get("split", {})
    manifest_path = data_cfg.get("manifest_path")
    dataset_root = data_cfg.get("dataset_root")

    if manifest_path:
        samples = load_manifest(manifest_path, dataset_root=dataset_root)
    else:
        samples = [{"id": f"s{i}", "split": "test"} for i in range(len(dataset))]

    ratios = split_cfg.get("ratios", {"train": 0.6, "val": 0.2, "test": 0.2})
    samples = assign_splits(
        samples,
        mode=split_cfg.get("mode", "sample"),
        seed=int(split_cfg.get("seed", 42)),
        ratios=ratios,
    )

    val_indices = [i for i, s in enumerate(samples) if s.get("split") == "val"]
    test_indices = [i for i, s in enumerate(samples) if s.get("split") == "test"]

    # Tiled prediction config
    ph, pw = infer_cfg.get("patch_size", [384, 384])
    sh, sw = infer_cfg.get("stride", [192, 192])

    def _tiled_predict_single(image: torch.Tensor) -> torch.Tensor:
        return tiled_predict_proba(
            model, image,
            patch_size=(int(ph), int(pw)),
            stride=(int(sh), int(sw)),
            blend=str(infer_cfg.get("blend", "hann")),
            pad_mode=str(infer_cfg.get("pad_mode", "reflect")),
            batch_tiles=int(infer_cfg.get("batch_tiles", 4)),
            device=device,
            eps=float(infer_cfg.get("eps", 1e-6)),
        )

    # Threshold sweep on val set
    print("  Threshold sweep on val set...")
    val_probs_list = []
    val_gt_list = []
    val_fov_list = []

    with torch.no_grad():
        for i, idx in enumerate(val_indices):
            sample = dataset[idx]
            image = sample["image"]

            if use_tta:
                if image.ndim == 3:
                    image = image.unsqueeze(0)
                probs = tta_predict_proba(
                    predict_fn=_tiled_predict_single,
                    image=image.to(device),
                    use_d4=True,
                    scales=[1.0],
                )
            else:
                probs = _tiled_predict_single(image)

            val_probs_list.append(probs.cpu())
            val_gt_list.append(sample["mask"].unsqueeze(0).cpu())
            if "fov" in sample:
                val_fov_list.append(sample["fov"].unsqueeze(0).unsqueeze(0).cpu().float())
            else:
                val_fov_list.append(torch.ones(1, 1, probs.shape[-2], probs.shape[-1]))

            sid = sample.get("id", f"val_{idx}")
            print(f"    Val {i+1}/{len(val_indices)}: {sid}")

    val_probs = torch.cat(val_probs_list, dim=0)
    val_gt = torch.cat(val_gt_list, dim=0)
    val_fov = torch.cat(val_fov_list, dim=0)

    # Sweep thresholds
    grid = [round(0.1 + i * 0.05, 2) for i in range(17)]
    best_dice = -1.0
    tau_star = 0.5
    for tau in grid:
        pred = (val_probs >= tau).float() * val_fov
        d = float(dice_score(pred, val_gt * val_fov).item())
        if d > best_dice:
            best_dice = d
            tau_star = tau

    print(f"  τ* = {tau_star:.2f} (val Dice = {best_dice:.4f})")

    # Evaluate on test set with τ*
    print("  Evaluating test set...")
    test_probs_list = []
    test_gt_list = []
    test_fov_list = []

    with torch.no_grad():
        for i, idx in enumerate(test_indices):
            sample = dataset[idx]
            image = sample["image"]

            if use_tta:
                if image.ndim == 3:
                    image = image.unsqueeze(0)
                probs = tta_predict_proba(
                    predict_fn=_tiled_predict_single,
                    image=image.to(device),
                    use_d4=True,
                    scales=[1.0],
                )
            else:
                probs = _tiled_predict_single(image)

            test_probs_list.append(probs.cpu())
            test_gt_list.append(sample["mask"].unsqueeze(0).cpu())
            if "fov" in sample:
                test_fov_list.append(sample["fov"].unsqueeze(0).unsqueeze(0).cpu().float())
            else:
                test_fov_list.append(torch.ones(1, 1, probs.shape[-2], probs.shape[-1]))

            sid = sample.get("id", f"test_{idx}")
            print(f"    Test {i+1}/{len(test_indices)}: {sid}")

    test_probs = torch.cat(test_probs_list, dim=0)
    test_gt = torch.cat(test_gt_list, dim=0)
    test_fov = torch.cat(test_fov_list, dim=0)

    # Apply threshold + post-processing
    pred_binary = (test_probs >= tau_star).float()

    if use_postprocess:
        processed = []
        for b in range(pred_binary.shape[0]):
            p = postprocess_vessel_mask(
                pred_binary[b, 0],
                min_size=min_component_size,
                close_radius=close_radius,
            )
            processed.append(p.unsqueeze(0).unsqueeze(0))
        pred_binary = torch.cat(processed, dim=0)

    # Compute all metrics
    metrics = compute_all_metrics(pred_binary, test_probs, test_gt, test_fov)
    metrics["tau_star"] = tau_star
    metrics["val_dice"] = best_dice
    return metrics


# ─── Output Formatting ────────────────────────────────────────

def print_table(results: list[dict]) -> None:
    """Print formatted comparison table."""
    header = f"{'Configuration':<30} {'Dice':>7} {'AUC':>7} {'Se':>7} {'Sp':>7} {'clDice':>7} {'τ*':>5}"
    sep = "─" * len(header)

    print()
    print("╔" + "═" * (len(header) + 2) + "╗")
    print("║ FP-Swin-UNet Ablation Study — HRF Dataset" + " " * (len(header) - 42) + " ║")
    print("╠" + "═" * (len(header) + 2) + "╣")
    print("║ " + header + " ║")
    print("╠" + "═" * (len(header) + 2) + "╣")

    for r in results:
        if r.get("status") == "no_checkpoint":
            line = f"║ {r['name']:<30} {'—':>7} {'—':>7} {'—':>7} {'—':>7} {'—':>7} {'—':>5} ║"
        else:
            m = r["metrics"]
            line = (
                f"║ {r['name']:<30} "
                f"{m['Dice']:>7.4f} "
                f"{m['AUC']:>7.4f} "
                f"{m['Sensitivity']:>7.4f} "
                f"{m['Specificity']:>7.4f} "
                f"{m['clDice']:>7.4f} "
                f"{m['tau_star']:>5.2f} ║"
            )
        print(line)

    print("╚" + "═" * (len(header) + 2) + "╝")
    print()


def save_csv(results: list[dict], output_path: Path) -> None:
    """Save results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["config", "Dice", "AUC", "Sensitivity", "Specificity", "clDice", "tau_star", "val_dice"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            if r.get("status") == "no_checkpoint":
                writer.writerow({"config": r["name"]})
            else:
                row = {"config": r["name"]}
                row.update(r["metrics"])
                writer.writerow(row)

    print(f"Results saved to {output_path}")


def save_json(results: list[dict], output_path: Path) -> None:
    """Save results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clean = []
    for r in results:
        entry = {"name": r["name"], "config_path": r.get("config_path", "")}
        if "metrics" in r:
            entry["metrics"] = r["metrics"]
        clean.append(entry)

    with open(output_path, "w") as f:
        json.dump(clean, f, indent=2)

    print(f"Results saved to {output_path}")


# ─── Main ─────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation evaluation with TTA and metrics.")
    parser.add_argument("--ablation-dir", type=str, default="runs", help="Base directory for run outputs.")
    parser.add_argument("--output", type=str, default="results/ablation_table.csv", help="CSV output path.")
    parser.add_argument("--no-tta", action="store_true", help="Disable TTA.")
    parser.add_argument("--no-postprocess", action="store_true", help="Disable post-processing.")
    parser.add_argument("--config", type=str, default=None, help="Evaluate a single config file (skip ablation dir scan).")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path (required with --config).")
    args = parser.parse_args()

    device = resolve_device()
    print(f"Device: {device}")

    results = []

    if args.config:
        # Single config mode
        if not args.checkpoint:
            print("ERROR: --checkpoint required with --config")
            sys.exit(1)
        name = Path(args.config).stem
        print(f"\nEvaluating: {name}")
        metrics = evaluate_config(
            args.config, args.checkpoint, device,
            use_tta=not args.no_tta,
            use_postprocess=not args.no_postprocess,
        )
        results.append({"name": name, "config_path": args.config, "metrics": metrics})
    else:
        # Ablation scan mode
        runs = find_ablation_runs(Path(args.ablation_dir))
        if not runs:
            print("No ablation runs found. Train configs first.")
            sys.exit(1)

        for run in runs:
            print(f"\n{'='*60}")
            print(f"Config: {run['name']}")
            print(f"{'='*60}")

            if run["status"] == "no_checkpoint":
                print("  ⚠ No checkpoint found — skipping")
                results.append(run)
                continue

            metrics = evaluate_config(
                run["config_path"], run["checkpoint"], device,
                use_tta=not args.no_tta,
                use_postprocess=not args.no_postprocess,
            )
            run["metrics"] = metrics
            results.append(run)

    # Output
    print_table(results)
    output_path = Path(args.output)
    save_csv(results, output_path)
    save_json(results, output_path.with_suffix(".json"))


if __name__ == "__main__":
    main()
