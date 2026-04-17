"""Inference entrypoint for Fractal-Prior Swin-UNet."""

from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image

from .config import apply_override, load_config
from .data import InfiniteRandomPatchDataset
from .data.registry import build_dataset
from .device import resolve_device
from .fractal import LFDParams, compute_lfd_map
from .inference.tiling import tiled_predict_proba
from .model import SimpleSwinUNet
from .model_factory import build_model as _build_model_from_cfg, resolve_in_channels
from .models import FractalPriorSwinUNet, SwinUNetTiny
from .seed import set_deterministic_seed
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
)
from .exp.split import assign_splits


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
        maps = [provider.get_patch(f"infer_{i}", image_patch=sample) for i, sample in enumerate(x)]
        return torch.stack(maps, dim=0)
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


def _load_tau_from_run(run_dir: Path) -> float | None:
    threshold_path = run_dir / "threshold.json"
    if threshold_path.exists():
        payload = json.loads(threshold_path.read_text(encoding="utf-8"))
        return float(payload.get("tau_star"))
    return None


def _save_png_masks(probs: torch.Tensor, tau: float, out_dir: Path, save_probs: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = (probs >= tau).float().cpu().numpy()
    for idx in range(preds.shape[0]):
        mask = (preds[idx, 0] * 255).astype("uint8")
        Image.fromarray(mask).save(out_dir / f"pred_{idx}.png")
        if save_probs:
            prob = (probs[idx, 0].clamp(0, 1) * 255).byte().cpu().numpy()
            Image.fromarray(prob).save(out_dir / f"prob_{idx}.png")


def infer(cfg: dict[str, Any], use_synth_data: bool, checkpoint: str | None, run_dir: Path, tau: float) -> None:
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

    infer_cfg = cfg.get("infer", {})
    mode = infer_cfg.get("mode", "patch")
    tiled_cfg = infer_cfg.get("tiled", {})
    save_probs = bool(infer_cfg.get("save_probs", True))
    out_subdir = infer_cfg.get("out_subdir", "preds")

    infer_samples = [s for s in samples if s.get("split") == "test"] or samples

    data_cfg = cfg.get("data", {})
    dataset_name = data_cfg.get("dataset")

    if use_synth_data:
        batch = _make_synth_batch(infer_samples, cfg, seed, full_image=(mode == "tiled"))
        x = batch["image"].to(device)
    elif dataset_name == "hrf":
        dataset = build_dataset(cfg)
        test_indices = [i for i, s in enumerate(dataset.manifest) if s.get("split") == "test"]
        if not test_indices:
            test_indices = list(range(len(dataset)))
        images = [dataset[idx]["image"] for idx in test_indices]
        x = torch.stack(images, dim=0).to(device)
    else:
        raise ValueError("Only synthetic data or HRF dataset is supported in this scaffold.")

    with torch.no_grad():
        if mode == "tiled":
            ph, pw = tiled_cfg.get("patch_size", [96, 96])
            sh, sw = tiled_cfg.get("stride", [48, 48])
            # H3: Process ALL test images, not just x[0]
            probs_list = []
            for img_idx in range(x.shape[0]):
                prob = tiled_predict_proba(
                    model,
                    x[img_idx],
                    patch_size=(int(ph), int(pw)),
                    stride=(int(sh), int(sw)),
                    blend=str(tiled_cfg.get("blend", "hann")),
                    pad_mode=str(tiled_cfg.get("pad_mode", "reflect")),
                    batch_tiles=int(tiled_cfg.get("batch_tiles", 8)),
                    device=device,
                    eps=float(tiled_cfg.get("eps", 1e-6)),
                    fractal_prior_config=cfg.get("fractal_prior"),
                )
                probs_list.append(prob)
                print(f"  Inferred image {img_idx + 1}/{x.shape[0]}")
            probs = torch.cat(probs_list, dim=0)
        else:
            if isinstance(model, SwinUNetTiny):
                out = model(x)
            elif isinstance(model, FractalPriorSwinUNet):
                lfd_map = _compute_lfd_batch(x, lfd_params, model=model)
                out = model(x, lfd_map=lfd_map)
            else:
                lfd_map = _compute_lfd_batch(x, lfd_params, model=model)
                out = model(x, lfd_map=lfd_map)
            logits = out["logits"] if isinstance(out, dict) else out
            probs = torch.sigmoid(logits)

    preds_dir = run_dir / out_subdir
    _save_png_masks(probs, tau, preds_dir, save_probs=save_probs)
    torch.save({"probs": probs.cpu()}, run_dir / "preds.pt")

    metrics = {"tau": tau, "num_samples": int(probs.shape[0]), "mode": mode}
    write_metrics(run_dir, metrics)
    print(f"Saved predictions to {preds_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer with Fractal-Prior Swin-UNet.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--use_synth_data", action="store_true", help="Use synthetic data.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output base directory.")
    parser.add_argument("--run_id", type=str, default=None, help="Run identifier.")
    parser.add_argument("--threshold_run_dir", type=str, default=None, help="Run dir containing threshold.json.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override config: key=value")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for override in args.overrides:
        apply_override(cfg, override)

    run_dir = make_run_dir(base=args.out_dir, run_id=args.run_id)
    cfg.setdefault("run", {})["id"] = run_dir.name
    cfg["run"]["base"] = str(run_dir.parent)

    write_resolved_config(run_dir, cfg)
    write_env(run_dir)
    write_git_commit(run_dir)
    write_code_hash(run_dir)
    write_readme(run_dir, " ".join(sys.argv))

    checkpoint = args.checkpoint or cfg.get("eval", {}).get("checkpoint_path")

    tau_cfg = cfg.get("eval", {}).get("threshold", {})
    tau = float(tau_cfg.get("default", 0.5))
    if tau_cfg.get("use_tau_star_if_available", False) and args.threshold_run_dir:
        tau_from_run = _load_tau_from_run(Path(args.threshold_run_dir))
        if tau_from_run is not None:
            tau = tau_from_run

    infer(cfg, use_synth_data=args.use_synth_data, checkpoint=checkpoint, run_dir=run_dir, tau=tau)


if __name__ == "__main__":
    main()
