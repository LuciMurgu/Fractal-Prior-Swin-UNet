#!/usr/bin/env python3
"""Quick clDice evaluation for cross-dataset checkpoints.

Loads the best checkpoint from each xdataset run, evaluates on a limited
number of FIVES val+test images, and reports Dice + clDice side by side.
"""
import json, sys, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fractal_swin_unet.config import load_config
from fractal_swin_unet.data.registry import build_dataset
from fractal_swin_unet.device import resolve_device
from fractal_swin_unet.exp.split import assign_splits
from fractal_swin_unet.inference.tiling import tiled_predict_proba
from fractal_swin_unet.metrics import cldice_score, dice_score
from fractal_swin_unet.model_factory import build_model
from fractal_swin_unet.seed import set_deterministic_seed

MAX_SAMPLES = 20  # Limit per split for speed


def eval_checkpoint(run_name: str, ckpt_path: str, config_path: str):
    cfg = load_config(config_path)
    set_deterministic_seed(42)
    device = resolve_device()

    model = build_model(cfg).to(device)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()

    dataset = build_dataset(cfg)
    split_cfg = cfg.get("data", {}).get("split", {})
    mode = split_cfg.get("mode", "sample")
    seed = int(split_cfg.get("seed", 42))
    ratios = split_cfg.get("ratios", {"train": 0.8, "val": 0.1, "test": 0.1})
    manifest = assign_splits(dataset.manifest, mode=mode, seed=seed, ratios=ratios)
    dataset.manifest = manifest

    tiled_cfg = cfg.get("infer", {}).get("tiled", {})
    ph, pw = tiled_cfg.get("patch_size", [384, 384])
    sh, sw = tiled_cfg.get("stride", [192, 192])
    fractal_cfg = cfg.get("fractal_prior", {})

    results = {}
    for split_name in ["val", "test"]:
        indices = [i for i, s in enumerate(manifest) if s.get("split") == split_name]
        indices = indices[:MAX_SAMPLES]
        if not indices:
            continue

        probs_list, gts_list = [], []
        with torch.no_grad():
            for j, idx in enumerate(indices):
                sample = dataset[idx]
                image = sample["image"]
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
                gts_list.append(sample["mask"].unsqueeze(0).cpu())
                print(f"  [{run_name}] {split_name} {j+1}/{len(indices)}", end="\r")

        all_probs = torch.cat(probs_list, dim=0)
        all_gts = torch.cat(gts_list, dim=0)

        # Sweep tau for best dice
        best_dice, best_tau = 0.0, 0.5
        for tau in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
            d = float(dice_score((all_probs >= tau).float(), all_gts).item())
            if d > best_dice:
                best_dice, best_tau = d, tau

        preds = (all_probs >= best_tau).float()
        cl = float(cldice_score(preds, all_gts).item())
        results[split_name] = {
            "dice": best_dice,
            "cldice": cl,
            "tau_star": best_tau,
            "n_samples": len(indices),
        }
        print(f"  [{run_name}] {split_name}: Dice={best_dice:.4f} clDice={cl:.4f} τ*={best_tau:.2f} (n={len(indices)})")

    return results


if __name__ == "__main__":
    runs = [
        ("Baseline", "runs/xdataset_fives_baseline/checkpoint_best.pt", "configs/cross_dataset/fives_baseline.yaml"),
        ("Fractal", "runs/xdataset_fives_fractal/checkpoint_best.pt", "configs/cross_dataset/fives_fractal.yaml"),
    ]

    all_results = {}
    for name, ckpt, config in runs:
        if not Path(ckpt).exists():
            print(f"SKIP {name}: {ckpt} not found")
            continue
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        all_results[name] = eval_checkpoint(name, ckpt, config)
        # Cleanup GPU
        import gc; gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Run':<20} {'Split':<6} {'Dice':>8} {'clDice':>8} {'τ*':>6} {'n':>4}")
    print(f"{'-'*80}")
    for name, splits in all_results.items():
        for split, m in splits.items():
            print(f"{name:<20} {split:<6} {m['dice']:>8.4f} {m['cldice']:>8.4f} {m['tau_star']:>6.2f} {m['n_samples']:>4}")
    print(f"{'='*80}")
