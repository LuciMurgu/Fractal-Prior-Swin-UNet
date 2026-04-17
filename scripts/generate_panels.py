#!/usr/bin/env python3
"""Generate qualitative comparison panels for the paper.

For each dataset, picks representative images and creates side-by-side panels:
  [Original | Ground Truth | Baseline Pred | Fractal Pred | Difference]

Difference panel highlights: green=fractal-only, red=baseline-only, yellow=both.

Usage:
    python scripts/generate_panels.py
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fractal_swin_unet.config import load_config
from fractal_swin_unet.data.registry import build_dataset
from fractal_swin_unet.device import resolve_device
from fractal_swin_unet.exp.split import assign_splits
from fractal_swin_unet.inference.tiling import tiled_predict_proba
from fractal_swin_unet.model_factory import build_model
from fractal_swin_unet.seed import set_deterministic_seed

OUTPUT_DIR = Path("reports/evidence_pack/panels")

# Dataset configs with specific image indices to visualize
PANEL_CONFIGS = [
    {
        "dataset": "FIVES",
        "config": "configs/cross_dataset/fives_baseline.yaml",
        "manifest": "manifests/fives.jsonl",
        "root": "/home/lucian/medical-imaging-ai/Retina Datasets/Full retina dataset/FIVES",
        "fov": False,
        "indices": [5, 15],  # Will pick from val/test
        "split": {"mode": "sample", "seed": 123, "ratios": {"train": 0.8, "val": 0.1, "test": 0.1}},
        "tau_baseline": 0.55,
        "tau_fractal": 0.50,
    },
    {
        "dataset": "CHASE_DB1",
        "config": "configs/cross_dataset/fives_baseline.yaml",
        "manifest": "manifests/chase.jsonl",
        "root": "/home/lucian/medical-imaging-ai/Retina Datasets/Full retina dataset/CHASE",
        "fov": False,
        "indices": [0, 5],
        "split": None,
        "tau_baseline": 0.90,
        "tau_fractal": 0.80,
    },
    {
        "dataset": "STARE",
        "config": "configs/cross_dataset/fives_baseline.yaml",
        "manifest": "manifests/stare.jsonl",
        "root": "/home/lucian/medical-imaging-ai/Retina Datasets/Full retina dataset/STARE",
        "fov": False,
        "indices": [0, 3],
        "split": None,
        "tau_baseline": 0.90,
        "tau_fractal": 0.90,
    },
    {
        "dataset": "HRF",
        "config": "configs/cross_dataset/fives_baseline.yaml",
        "manifest": "manifests/hrf_full.jsonl",
        "root": "/home/lucian/medical-imaging-ai/Retina Datasets/Full retina dataset/HRF",
        "fov": True,
        "indices": [0, 5],
        "split": None,
        "tau_baseline": 0.90,
        "tau_fractal": 0.90,
    },
]

CHECKPOINTS = {
    "baseline": "runs/xdataset_fives_baseline/checkpoint_best.pt",
    "fractal": "runs/xdataset_fives_fractal/checkpoint_best.pt",
}

CONFIGS = {
    "baseline": "configs/cross_dataset/fives_baseline.yaml",
    "fractal": "configs/cross_dataset/fives_fractal.yaml",
}


def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model(model_name: str, device: torch.device):
    """Load a model checkpoint."""
    cfg = load_config(CONFIGS[model_name])
    model = build_model(cfg).to(device)
    state = torch.load(CHECKPOINTS[model_name], map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()
    del state
    cleanup_gpu()
    return model, cfg


def predict_image(model, image: torch.Tensor, cfg: dict, device: torch.device) -> torch.Tensor:
    """Run tiled inference on a single image."""
    tiled_cfg = cfg.get("infer", {}).get("tiled", {})
    ph, pw = tiled_cfg.get("patch_size", [384, 384])
    sh, sw = tiled_cfg.get("stride", [192, 192])
    fractal_cfg = cfg.get("fractal_prior", {})

    with torch.no_grad():
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
    return probs.cpu()


def tensor_to_rgb(t: torch.Tensor, max_size: int = 800) -> np.ndarray:
    """Convert image tensor (C,H,W) in [0,1] to RGB uint8, resized."""
    if t.shape[0] == 3:
        arr = (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    else:
        arr = (t.squeeze().numpy() * 255).clip(0, 255).astype(np.uint8)
        arr = np.stack([arr, arr, arr], axis=2)

    h, w = arr.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = Image.fromarray(arr).resize((new_w, new_h), Image.LANCZOS)
        arr = np.array(img)
    return arr


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Binary mask to white-on-black RGB."""
    m = (mask * 255).astype(np.uint8)
    return np.stack([m, m, m], axis=2)


def make_diff_panel(gt: np.ndarray, baseline: np.ndarray, fractal: np.ndarray) -> np.ndarray:
    """Create difference panel:
    Green = fractal correct, baseline wrong (fractal improvement)
    Red = baseline correct, fractal wrong (fractal regression)
    Yellow = both correct
    Dark = both wrong or both correct negative
    """
    h, w = gt.shape
    panel = np.zeros((h, w, 3), dtype=np.uint8)

    # Background: dark gray
    panel[:, :] = [30, 30, 30]

    # Both correct (true positive for both) = dim white
    both_tp = gt & baseline & fractal
    panel[both_tp] = [100, 100, 100]

    # Fractal-only correct = bright green
    fractal_only = gt & fractal & ~baseline
    panel[fractal_only] = [0, 255, 80]

    # Baseline-only correct = red
    baseline_only = gt & baseline & ~fractal
    panel[baseline_only] = [255, 60, 60]

    # Both false positive (no GT but both predict) = dim yellow
    both_fp = ~gt & baseline & fractal
    panel[both_fp] = [60, 60, 0]

    # Fractal-only false positive = dim green
    fract_fp = ~gt & fractal & ~baseline
    panel[fract_fp] = [0, 80, 30]

    # Baseline-only false positive = dim red
    base_fp = ~gt & baseline & ~fractal
    panel[base_fp] = [80, 30, 30]

    return panel


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    """Add text label at top of image."""
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    # Use default font (no external font needed)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    # Semi-transparent background for text
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([2, 2, tw + 8, th + 8], fill=(0, 0, 0))
    draw.text((5, 3), label, fill=(255, 255, 255), font=font)
    return np.array(img)


def create_panel(
    image_rgb: np.ndarray,
    gt_mask: np.ndarray,
    baseline_mask: np.ndarray,
    fractal_mask: np.ndarray,
    sample_id: str,
    dataset: str,
    max_size: int = 800,
) -> Image.Image:
    """Create a 5-column comparison panel."""
    h, w = image_rgb.shape[:2]

    # Resize all to same dimensions
    def resize(arr):
        if arr.shape[:2] != (h, w):
            return np.array(Image.fromarray(arr).resize((w, h), Image.NEAREST))
        return arr

    gt_rgb = resize(mask_to_rgb(gt_mask))
    base_rgb = resize(mask_to_rgb(baseline_mask))
    frac_rgb = resize(mask_to_rgb(fractal_mask))
    diff_rgb = resize(make_diff_panel(gt_mask, baseline_mask, fractal_mask))

    # Add labels
    image_labeled = add_label(image_rgb, f"{dataset}: {sample_id}")
    gt_labeled = add_label(gt_rgb, "Ground Truth")
    base_labeled = add_label(base_rgb, "Baseline")
    frac_labeled = add_label(frac_rgb, "Fractal")
    diff_labeled = add_label(diff_rgb, "Diff (green=fractal+)")

    # Concatenate horizontally with 2px separator
    sep = np.ones((h, 2, 3), dtype=np.uint8) * 128
    panel = np.concatenate([
        image_labeled, sep, gt_labeled, sep, base_labeled, sep, frac_labeled, sep, diff_labeled
    ], axis=1)

    return Image.fromarray(panel)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = resolve_device()
    set_deterministic_seed(42)

    print(f"Device: {device}")
    print(f"Output: {OUTPUT_DIR}")

    # Load both models
    print("\nLoading baseline model...")
    model_base, cfg_base = load_model("baseline", device)

    print("Loading fractal model...")
    model_frac, cfg_frac = load_model("fractal", device)

    total_panels = 0

    for pcfg in PANEL_CONFIGS:
        ds_name = pcfg["dataset"]
        print(f"\n{'─'*60}")
        print(f"  {ds_name}")
        print(f"{'─'*60}")

        # Build dataset
        cfg = load_config(pcfg["config"])
        cfg["data"]["manifest_path"] = pcfg["manifest"]
        cfg["data"]["dataset_root"] = pcfg["root"]
        cfg["eval"]["fov"]["enabled"] = pcfg["fov"]

        dataset = build_dataset(cfg)

        # Assign splits if needed
        if pcfg.get("split"):
            s = pcfg["split"]
            dataset.manifest = assign_splits(
                dataset.manifest, mode=s["mode"], seed=s["seed"], ratios=s["ratios"]
            )
            # Use val/test only
            indices = [i for i, m in enumerate(dataset.manifest) if m.get("split") in ("val", "test")]
        else:
            indices = list(range(len(dataset)))

        # Pick specific indices
        pick = [indices[i] for i in pcfg["indices"] if i < len(indices)]

        for idx in pick:
            sample = dataset[idx]
            sid = sample.get("id", f"sample_{idx}")
            print(f"  Processing {sid}...")

            image = sample["image"]
            gt = sample["mask"].squeeze().numpy() > 0.5

            # Predict with both models
            probs_base = predict_image(model_base, image, cfg_base, device)
            probs_frac = predict_image(model_frac, image, cfg_frac, device)

            # Threshold
            pred_base = (probs_base.squeeze().numpy() >= pcfg["tau_baseline"])
            pred_frac = (probs_frac.squeeze().numpy() >= pcfg["tau_fractal"])

            # Convert image to RGB
            image_rgb = tensor_to_rgb(image, max_size=800)
            h, w = image_rgb.shape[:2]

            # Resize masks to match
            def resize_mask(m):
                return np.array(Image.fromarray(m.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 127

            gt_r = resize_mask(gt)
            base_r = resize_mask(pred_base)
            frac_r = resize_mask(pred_frac)

            # Create panel
            panel = create_panel(image_rgb, gt_r, base_r, frac_r, sid, ds_name)
            out_path = OUTPUT_DIR / f"{ds_name}_{sid}.png"
            panel.save(out_path)
            print(f"    ✅ Saved: {out_path}")
            total_panels += 1

        cleanup_gpu()

    print(f"\n{'='*60}")
    print(f"  DONE: {total_panels} panels saved to {OUTPUT_DIR}")
    print(f"{'='*60}")

    # Cleanup
    del model_base, model_frac
    cleanup_gpu()
    return 0


if __name__ == "__main__":
    sys.exit(main())
