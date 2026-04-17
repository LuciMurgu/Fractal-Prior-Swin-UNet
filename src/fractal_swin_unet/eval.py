"""Evaluation entrypoint for Fractal-Prior Swin-UNet."""

from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List

import torch

from .config import apply_override, load_config
from .data import InfiniteRandomPatchDataset
from .data.fov import load_fov_mask
from .data.registry import build_dataset
from .device import resolve_device
from .fractal import LFDParams, compute_lfd_map
from .inference.tiling import tiled_predict_proba
from .inference.tta import tta_predict_proba
from .inference.postprocess import postprocess_vessel_mask
from .losses import dice_bce_loss
from .metrics import auroc, cldice_score, dice_score, f1_score, pixel_accuracy, sensitivity, specificity
from .model import SimpleSwinUNet
from .model_factory import build_model as _build_model_from_cfg, resolve_in_channels
from .models import FractalPriorSwinUNet, SwinUNetTiny
from .seed import set_deterministic_seed
from .exp.audit import assert_no_patient_leakage
from .exp.manifest import load_manifest
from .exp.run_artifacts import (
    make_run_dir,
    write_code_hash,
    write_env,
    write_git_commit,
    write_manifest_and_hash,
    write_metrics,
    write_readme,
    write_resolved_config,
    write_threshold,
)
from .exp.split import assign_splits
from .exp.threshold import sweep_thresholds


# Config loading and overrides are in config.py
_load_config = load_config
_apply_override = apply_override


def _build_model(cfg: dict[str, Any]) -> torch.nn.Module:
    """Build model from config — delegates to shared model_factory."""
    return _build_model_from_cfg(cfg)


def _compute_lfd_batch(x: torch.Tensor, params: LFDParams, model: torch.nn.Module | None = None) -> torch.Tensor:
    prior_channels = getattr(model, 'prior_channels', 1) if model is not None else 1
    if prior_channels > 1:
        from .fractal.provider import FractalPriorConfig, FractalPriorProvider
        config = FractalPriorConfig(
            enabled=True,
            multi_scale_enabled=True,
            multi_scale_factors=(1, 2, 4),
            include_lacunarity=(prior_channels >= 4),
        )
        provider = FractalPriorProvider(config)
        maps = [provider.get_patch(f"eval_{i}", image_patch=sample) for i, sample in enumerate(x)]
        return torch.stack(maps, dim=0)  # (B, C, H, W)
    lfd_maps = [compute_lfd_map(sample, params=params) for sample in x]
    return torch.stack(lfd_maps, dim=0).unsqueeze(1)


def _stable_id_seed(sample_id: str, base_seed: int) -> int:
    digest = sha256(sample_id.encode("utf-8")).hexdigest()
    return base_seed + int(digest[:8], 16)


def _resolve_in_channels(cfg: dict[str, Any]) -> int:
    """Resolve input channels — delegates to shared model_factory."""
    return resolve_in_channels(cfg)


def _make_synth_patch(sample_id: str, cfg: dict[str, Any], seed: int, full_image: bool) -> dict[str, torch.Tensor]:
    data_cfg = cfg.get("data", {})
    image_size = int(data_cfg.get("image_size", 128))
    if full_image:
        image_size = int(data_cfg.get("full_image_size", image_size))
    in_channels = _resolve_in_channels(cfg)
    ps = data_cfg.get("patch_size", 96)
    patch_size = ps[0] if isinstance(ps, (list, tuple)) else int(ps)

    gen = torch.Generator().manual_seed(seed)
    image = torch.randn(in_channels, image_size, image_size, generator=gen)
    mask = (torch.randn(1, image_size, image_size, generator=gen) > 0).float()

    if full_image:
        return {"image": image, "mask": mask}

    dataset = InfiniteRandomPatchDataset(image=image, mask=mask, patch_size=patch_size, seed=seed)
    return next(iter(dataset))


def _make_synth_batch(samples: List[Dict[str, Any]], cfg: dict[str, Any], seed: int, full_image: bool) -> dict[str, torch.Tensor]:
    images = []
    masks = []
    for sample in samples:
        sample_id = str(sample["id"])
        sample_seed = _stable_id_seed(sample_id, seed)
        batch = _make_synth_patch(sample_id, cfg, sample_seed, full_image)
        images.append(batch["image"])
        masks.append(batch["mask"])
    return {
        "image": torch.stack(images, dim=0),
        "mask": torch.stack(masks, dim=0),
    }


def _prepare_manifest(cfg: dict[str, Any], use_synth_data: bool) -> List[Dict[str, Any]]:
    data_cfg = cfg.get("data", {})
    manifest_path = data_cfg.get("manifest_path")
    dataset_root = data_cfg.get("dataset_root")

    if manifest_path:
        samples = load_manifest(manifest_path, dataset_root=dataset_root)
    elif use_synth_data:
        synth_cfg = data_cfg.get("synth", {})
        num_samples = int(synth_cfg.get("num_samples", 6))
        samples = [
            {
                "id": f"synth_{i}",
                "image_path": f"synthetic://image_{i}",
                "mask_path": f"synthetic://mask_{i}",
            }
            for i in range(num_samples)
        ]
    else:
        raise ValueError("manifest_path is required for non-synthetic data.")

    split_cfg = data_cfg.get("split", {})
    locked = bool(split_cfg.get("locked", False))
    has_split = all("split" in s for s in samples)

    if locked and has_split:
        return samples

    mode = split_cfg.get("mode", "sample")
    seed = int(split_cfg.get("seed", 42))
    ratios = split_cfg.get("ratios", {"train": 0.8, "val": 0.1, "test": 0.1})
    return assign_splits(samples, mode=mode, seed=seed, ratios=ratios)


def _get_threshold_grid(cfg: dict[str, Any]) -> List[float]:
    sweep_cfg = cfg.get("eval", {}).get("threshold_sweep", {})
    grid = sweep_cfg.get("grid")
    if isinstance(grid, dict):
        start = float(grid.get("start", 0.1))
        stop = float(grid.get("stop", 0.9))
        step = float(grid.get("step", 0.05))
        values = []
        current = start
        while current <= stop + 1e-9:
            values.append(round(current, 6))
            current += step
        return values
    if isinstance(grid, list):
        return [float(x) for x in grid]
    return [round(0.1 + i * 0.05, 2) for i in range(17)]


def evaluate(cfg: dict[str, Any], use_synth_data: bool, checkpoint: str | None, run_dir: Path) -> None:
    seed = int(cfg.get("seed", 42))
    set_deterministic_seed(seed)
    device = resolve_device()

    model = _build_model(cfg).to(device)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])

    model.eval()
    lfd_params = LFDParams()

    samples = _prepare_manifest(cfg, use_synth_data=use_synth_data)
    write_manifest_and_hash(run_dir, samples)

    split_cfg = cfg.get("data", {}).get("split", {})
    allow_leakage = bool(split_cfg.get("allow_leakage", False))
    if not allow_leakage:
        assert_no_patient_leakage(samples)

    val_samples = [s for s in samples if s.get("split") == "val"]
    test_samples = [s for s in samples if s.get("split") == "test"]

    eval_cfg = cfg.get("eval", {})
    full_image = bool(eval_cfg.get("full_image", False))
    fov_cfg = eval_cfg.get("fov", {})
    use_fov = bool(fov_cfg.get("enabled", False))

    def _compute_probs(x: torch.Tensor) -> torch.Tensor:
        if full_image:
            tiled_cfg = cfg.get("infer", {}).get("tiled", {})
            ph, pw = tiled_cfg.get("patch_size", [96, 96])
            sh, sw = tiled_cfg.get("stride", [48, 48])
            probs_per_sample = []
            for image in x:
                probs_per_sample.append(
                    tiled_predict_proba(
                        model,
                        image,
                        patch_size=(int(ph), int(pw)),
                        stride=(int(sh), int(sw)),
                        blend=str(tiled_cfg.get("blend", "hann")),
                        pad_mode=str(tiled_cfg.get("pad_mode", "reflect")),
                        batch_tiles=int(tiled_cfg.get("batch_tiles", 8)),
                        device=device,
                        eps=float(tiled_cfg.get("eps", 1e-6)),
                        fractal_prior_config=cfg.get("fractal_prior"),
                    )
                )
            return torch.cat(probs_per_sample, dim=0)
        if isinstance(model, SwinUNetTiny):
            out = model(x)
        elif isinstance(model, FractalPriorSwinUNet):
            lfd_map = _compute_lfd_batch(x, lfd_params, model=model)
            out = model(x, lfd_map=lfd_map)
        else:
            lfd_map = _compute_lfd_batch(x, lfd_params, model=model)
            out = model(x, lfd_map=lfd_map)
        logits = out["logits"] if isinstance(out, dict) else out
        return torch.sigmoid(logits)

    data_cfg = cfg.get("data", {})
    dataset_name = data_cfg.get("dataset")

    def _collect_full_image_split(
        dataset_obj: Any, indices: List[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        probs_list: List[torch.Tensor] = []
        masks_list: List[torch.Tensor] = []
        fov_list: List[torch.Tensor] = []
        tiled_cfg = cfg.get("infer", {}).get("tiled", {})
        ph, pw = tiled_cfg.get("patch_size", [96, 96])
        sh, sw = tiled_cfg.get("stride", [48, 48])

        # TTA configuration
        tta_cfg = eval_cfg.get("tta", {})
        tta_enabled = bool(tta_cfg.get("enabled", False))
        tta_scales = tta_cfg.get("scales", [1.0])

        def _tiled_predict_single(image: torch.Tensor) -> torch.Tensor:
            """Run tiled prediction on a single image, returning probs."""
            return tiled_predict_proba(
                model,
                image,
                patch_size=(int(ph), int(pw)),
                stride=(int(sh), int(sw)),
                blend=str(tiled_cfg.get("blend", "hann")),
                pad_mode=str(tiled_cfg.get("pad_mode", "reflect")),
                batch_tiles=int(tiled_cfg.get("batch_tiles", 8)),
                device=device,
                eps=float(tiled_cfg.get("eps", 1e-6)),
                fractal_prior_config=cfg.get("fractal_prior"),
            )

        with torch.no_grad():
            for i, idx in enumerate(indices):
                sample = dataset_obj[idx]
                image = sample["image"]

                if tta_enabled:
                    # TTA wraps the tiled prediction
                    if image.ndim == 3:
                        image = image.unsqueeze(0)
                    probs = tta_predict_proba(
                        predict_fn=_tiled_predict_single,
                        image=image.to(device),
                        use_d4=True,
                        scales=tta_scales,
                    )
                else:
                    probs = _tiled_predict_single(image)

                probs_list.append(probs.cpu())
                mask = sample["mask"].unsqueeze(0).cpu()
                masks_list.append(mask)
                sample_id = sample.get("id", f"sample_{idx}")
                print(f"  Eval image {i+1}/{len(indices)}: {sample_id}"
                      f"{' [TTA]' if tta_enabled else ''}")
                if use_fov and "fov" in sample:
                    fov_list.append(sample["fov"].unsqueeze(0).unsqueeze(0).cpu().float())
                else:
                    fov_list.append(torch.ones_like(mask).cpu())
        probs_cat = torch.cat(probs_list, dim=0)
        masks_cat = torch.cat(masks_list, dim=0)
        if use_fov:
            return probs_cat, masks_cat, torch.cat(fov_list, dim=0)
        return probs_cat, masks_cat, None

    if use_synth_data:
        val_source = val_samples or samples[:1]
        val_batch = _make_synth_batch(val_source, cfg, seed + 1, full_image)
        x_val = val_batch["image"].to(device)
        y_val = val_batch["mask"].to(device)
        fov_mask = None
    elif dataset_name == "hrf":
        dataset = build_dataset(cfg)
        val_indices = [i for i, s in enumerate(dataset.manifest) if s.get("split") == "val"]
        if not val_indices:
            val_indices = list(range(len(dataset)))
        if full_image:
            x_val = None
            val_probs, y_val, fov_mask = _collect_full_image_split(dataset, val_indices)
        else:
            images = []
            masks = []
            fovs = []
            for idx in val_indices:
                sample = dataset[idx]
                images.append(sample["image"])
                masks.append(sample["mask"])
                if use_fov and "fov" in sample:
                    fovs.append(sample["fov"])
                else:
                    fovs.append(torch.ones_like(sample["mask"]))
            x_val = torch.stack(images, dim=0).to(device)
            y_val = torch.stack(masks, dim=0).to(device)
            fov_mask = torch.stack(fovs, dim=0) if use_fov else None
    else:
        raise ValueError("Only synthetic data or HRF dataset is supported in this scaffold.")

    with torch.no_grad():
        if dataset_name == "hrf" and full_image and not use_synth_data:
            val_loss = dice_bce_loss(val_probs.logit(), y_val)
        else:
            val_probs = _compute_probs(x_val)
            val_loss = dice_bce_loss(val_probs.logit(), y_val)

    if use_fov and use_synth_data:
        fov_mask = (torch.ones_like(y_val[0, 0]) > 0)

    fov_mask_4d = None
    if fov_mask is not None:
        fov_mask_4d = fov_mask
        if fov_mask_4d.ndim == 2:
            fov_mask_4d = fov_mask_4d.unsqueeze(0).unsqueeze(0)
        elif fov_mask_4d.ndim == 3:
            fov_mask_4d = fov_mask_4d.unsqueeze(1)

    metrics: Dict[str, Any] = {"val_loss": float(val_loss.item())}
    grid = _get_threshold_grid(cfg)
    if fov_mask_4d is not None:
        val_probs = val_probs * fov_mask_4d.to(val_probs.device)
        y_val = y_val * fov_mask_4d.to(y_val.device)
    sweep = sweep_thresholds(val_probs, y_val, grid)
    tau_star = sweep["tau_star"]
    metrics.update(
        {
            "best_dice": sweep["best_dice"],
            "dice_at_0_5": sweep["dice_at_0_5"],
            "best_cldice": sweep["best_cldice"],
            "cldice_at_0_5": sweep["cldice_at_0_5"],
            "tau_star": tau_star,
            "auc_roc": sweep["auc_roc"],
            "se_tau_star": sweep["se_tau_star"],
            "sp_tau_star": sweep["sp_tau_star"],
            "acc_tau_star": sweep["acc_tau_star"],
            "f1_tau_star": sweep["f1_tau_star"],
            "se_at_0_5": sweep["se_at_0_5"],
            "sp_at_0_5": sweep["sp_at_0_5"],
            "acc_at_0_5": sweep["acc_at_0_5"],
            "f1_at_0_5": sweep["f1_at_0_5"],
        }
    )
    write_threshold(run_dir, sweep)
    print(f"BestDice {sweep['best_dice']:.4f} BestclDice {sweep['best_cldice']:.4f} "
          f"AUC {sweep['auc_roc']:.4f} Se {sweep['se_tau_star']:.4f} Sp {sweep['sp_tau_star']:.4f} "
          f"Acc {sweep['acc_tau_star']:.4f} F1 {sweep['f1_tau_star']:.4f} tau* {tau_star:.2f}")

    if test_samples:
        if use_synth_data:
            test_batch = _make_synth_batch(test_samples, cfg, seed + 2, full_image)
            x_test = test_batch["image"].to(device)
            y_test = test_batch["mask"].to(device)
            test_fov = fov_mask
        elif dataset_name == "hrf":
            # M5: Reuse the dataset object built for val, don't rebuild
            test_indices = [i for i, s in enumerate(dataset.manifest) if s.get("split") == "test"]
            if not test_indices:
                test_indices = list(range(len(dataset)))
            if full_image:
                x_test = None
                test_probs, y_test, test_fov = _collect_full_image_split(dataset, test_indices)
            else:
                images = []
                masks = []
                fovs = []
                for idx in test_indices:
                    sample = dataset[idx]
                    images.append(sample["image"])
                    masks.append(sample["mask"])
                    if use_fov and "fov" in sample:
                        fovs.append(sample["fov"])
                x_test = torch.stack(images, dim=0).to(device)
                y_test = torch.stack(masks, dim=0).to(device)
                test_fov = torch.stack(fovs, dim=0) if fovs else None
        else:
            test_fov = None
            x_test = None
            y_test = None

        if x_test is not None:
            with torch.no_grad():
                test_probs = _compute_probs(x_test)
            tau_star = float(metrics["tau_star"])
            test_fov_mask = None
            if test_fov is not None:
                test_fov_mask = test_fov
                if test_fov_mask.ndim == 2:
                    test_fov_mask = test_fov_mask.unsqueeze(0).unsqueeze(0)
                elif test_fov_mask.ndim == 3:
                    test_fov_mask = test_fov_mask.unsqueeze(1)
                test_fov_mask = test_fov_mask.to(y_test.device)
            preds_tau = (test_probs >= tau_star).float()
            preds_05 = (test_probs >= 0.5).float()
            metrics["test_dice_tau_star"] = float(
                dice_score(preds_tau, y_test, mask=test_fov_mask).item()
            )
            metrics["test_dice_at_0_5"] = float(
                dice_score(preds_05, y_test, mask=test_fov_mask).item()
            )
            metrics["test_cldice_tau_star"] = float(
                cldice_score(preds_tau, y_test, mask=test_fov_mask).item()
            )
            metrics["test_cldice_at_0_5"] = float(
                cldice_score(preds_05, y_test, mask=test_fov_mask).item()
            )
            metrics["test_se_tau_star"] = float(sensitivity(preds_tau, y_test, mask=test_fov_mask).item())
            metrics["test_sp_tau_star"] = float(specificity(preds_tau, y_test, mask=test_fov_mask).item())
            metrics["test_acc_tau_star"] = float(pixel_accuracy(preds_tau, y_test, mask=test_fov_mask).item())
            metrics["test_f1_tau_star"] = float(f1_score(preds_tau, y_test, mask=test_fov_mask).item())
            metrics["test_auc_roc"] = auroc(test_probs, y_test, mask=test_fov_mask)
        elif dataset_name == "hrf" and full_image:
            tau_star = float(metrics["tau_star"])
            test_fov_mask = None
            if test_fov is not None:
                test_fov_mask = test_fov
                if test_fov_mask.ndim == 2:
                    test_fov_mask = test_fov_mask.unsqueeze(0).unsqueeze(0)
                elif test_fov_mask.ndim == 3:
                    test_fov_mask = test_fov_mask.unsqueeze(1)
                test_fov_mask = test_fov_mask.to(y_test.device)
            preds_tau = (test_probs >= tau_star).float()
            preds_05 = (test_probs >= 0.5).float()
            metrics["test_dice_tau_star"] = float(
                dice_score(preds_tau, y_test, mask=test_fov_mask).item()
            )
            metrics["test_dice_at_0_5"] = float(
                dice_score(preds_05, y_test, mask=test_fov_mask).item()
            )
            metrics["test_cldice_tau_star"] = float(
                cldice_score(preds_tau, y_test, mask=test_fov_mask).item()
            )
            metrics["test_cldice_at_0_5"] = float(
                cldice_score(preds_05, y_test, mask=test_fov_mask).item()
            )
            metrics["test_se_tau_star"] = float(sensitivity(preds_tau, y_test, mask=test_fov_mask).item())
            metrics["test_sp_tau_star"] = float(specificity(preds_tau, y_test, mask=test_fov_mask).item())
            metrics["test_acc_tau_star"] = float(pixel_accuracy(preds_tau, y_test, mask=test_fov_mask).item())
            metrics["test_f1_tau_star"] = float(f1_score(preds_tau, y_test, mask=test_fov_mask).item())
            metrics["test_auc_roc"] = auroc(test_probs, y_test, mask=test_fov_mask)

    write_metrics(run_dir, metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Fractal-Prior Swin-UNet.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--use_synth_data", action="store_true", help="Use synthetic data.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path.")
    parser.add_argument("--run_id", type=str, default=None, help="Run identifier.")
    parser.add_argument("--run_base", type=str, default="runs", help="Base directory for runs.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override config: key=value")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for override in args.overrides:
        apply_override(cfg, override)

    run_dir = make_run_dir(base=args.run_base, run_id=args.run_id)
    cfg.setdefault("run", {})["id"] = run_dir.name
    cfg["run"]["base"] = str(run_dir.parent)

    write_resolved_config(run_dir, cfg)
    write_env(run_dir)
    write_git_commit(run_dir)
    write_code_hash(run_dir)
    write_readme(run_dir, " ".join(sys.argv))

    checkpoint = args.checkpoint or cfg.get("eval", {}).get("checkpoint_path")
    evaluate(cfg, use_synth_data=args.use_synth_data, checkpoint=checkpoint, run_dir=run_dir)


if __name__ == "__main__":
    main()
