"""Training entrypoint for Fractal-Prior Swin-UNet."""

from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from .config import apply_override, load_config
from .data import InfiniteRandomPatchDataset
from .data.registry import build_dataset
from .fractal import LFDParams, compute_lfd_map
from .inference.tiling import tiled_predict_proba
from .losses import CompositeLoss
from .metrics import dice_score
from .model import SimpleSwinUNet
from .model_factory import build_model as _build_model_from_cfg, resolve_in_channels
from .models import FractalPriorSwinUNet, SwinUNetTiny
from .device import resolve_device
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
from .exp.threshold import sweep_thresholds


# Config loading and overrides are in config.py
_load_config = load_config
_apply_override = apply_override


def _build_model(cfg: dict[str, Any]) -> torch.nn.Module:
    """Build model from config — delegates to shared model_factory."""
    return _build_model_from_cfg(cfg)


def _compute_lfd_batch(x: torch.Tensor, params: LFDParams) -> torch.Tensor:
    lfd_maps = [compute_lfd_map(sample, params=params) for sample in x]
    return torch.stack(lfd_maps, dim=0).unsqueeze(1)


def _stable_id_seed(sample_id: str, base_seed: int) -> int:
    digest = sha256(sample_id.encode("utf-8")).hexdigest()
    return base_seed + int(digest[:8], 16)


def _resolve_in_channels(cfg: dict[str, Any]) -> int:
    """Resolve input channels — delegates to shared model_factory."""
    return resolve_in_channels(cfg)


def _make_synth_patch(sample_id: str, cfg: dict[str, Any], seed: int) -> dict[str, torch.Tensor]:
    data_cfg = cfg.get("data", {})
    image_size = int(data_cfg.get("image_size", 128))
    in_channels = _resolve_in_channels(cfg)
    ps = data_cfg.get("patch_size", 96)
    patch_size = ps[0] if isinstance(ps, (list, tuple)) else int(ps)

    gen = torch.Generator().manual_seed(seed)
    image = torch.randn(in_channels, image_size, image_size, generator=gen)
    mask = (torch.randn(1, image_size, image_size, generator=gen) > 0).float()

    dataset = InfiniteRandomPatchDataset(image=image, mask=mask, patch_size=patch_size, seed=seed)
    return next(iter(dataset))


def _make_synth_batch(samples: List[Dict[str, Any]], cfg: dict[str, Any], seed: int, batch_size: int) -> dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("No samples available for batch creation.")
    sample_id = str(samples[0]["id"])
    sample_seed = _stable_id_seed(sample_id, seed)
    batch = _make_synth_patch(sample_id, cfg, sample_seed)
    return {
        "image": batch["image"].unsqueeze(0).repeat(batch_size, 1, 1, 1),
        "mask": batch["mask"].unsqueeze(0).repeat(batch_size, 1, 1, 1),
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
    if isinstance(grid, list):
        return [float(x) for x in grid]
    if isinstance(grid, dict):
        start = float(grid.get("start", 0.1))
        stop = float(grid.get("stop", 0.9))
        step = float(grid.get("step", 0.05))
        values: List[float] = []
        cur = start
        while cur <= stop + 1e-9:
            values.append(round(cur, 6))
            cur += step
        return values
    return [round(0.1 + i * 0.05, 2) for i in range(17)]


def _forward_logits(
    model: torch.nn.Module,
    x: torch.Tensor,
    data_cfg: dict[str, Any],
    lfd_params: LFDParams,
    batch: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor | dict[str, torch.Tensor], torch.Tensor | None]:
    """Forward pass returning (output, lfd_map).

    Returns the lfd_map explicitly instead of storing it as a side effect
    on the model object (L2 cleanup).
    """
    if isinstance(model, SwinUNetTiny):
        return model(x), None
    if isinstance(model, FractalPriorSwinUNet):
        lfd_map = None if batch is None else batch.get("prior")
        if lfd_map is None:
            lfd_map = _compute_lfd_batch(x, lfd_params)
        else:
            lfd_map = lfd_map.to(x.device)
        out = model(x, lfd_map=lfd_map, expect_prior=bool(data_cfg.get("return_fractal_prior", False)))
        return out, lfd_map
    lfd_map = _compute_lfd_batch(x, lfd_params)
    return model(x, lfd_map=lfd_map), lfd_map


def _unpack_model_output(
    out: torch.Tensor | dict[str, torch.Tensor],
    targets: torch.Tensor,
    model: torch.nn.Module,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Unpack model output into logits + optional auxiliary loss.

    When model returns a dict (multi-decoder mode), extracts logits and
    computes auxiliary edge/skeleton/FD losses. Otherwise returns plain logits.
    """
    if isinstance(out, torch.Tensor):
        return out, None

    logits = out["logits"]
    aux_loss = torch.tensor(0.0, device=logits.device)
    loss_cfg = cfg.get("loss", {})

    # Edge auxiliary loss
    if "edge_logits" in out:
        edge_cfg = loss_cfg.get("edge_aux", {})
        if bool(edge_cfg.get("enabled", False)):
            from .models.multi_decoder import extract_edge_gt
            edge_gt = extract_edge_gt(targets)
            edge_w = float(edge_cfg.get("weight", 0.5))
            aux_loss = aux_loss + edge_w * F.binary_cross_entropy_with_logits(
                out["edge_logits"], edge_gt
            )

    # Skeleton auxiliary loss
    if "skeleton_logits" in out:
        skel_cfg = loss_cfg.get("skeleton_aux", {})
        if bool(skel_cfg.get("enabled", False)):
            from .models.multi_decoder import extract_skeleton_gt
            skel_gt = extract_skeleton_gt(targets, iters=8)
            skel_w = float(skel_cfg.get("weight", 0.5))
            aux_loss = aux_loss + skel_w * F.binary_cross_entropy_with_logits(
                out["skeleton_logits"], skel_gt
            )

    # FD regression loss
    if "fd_pred" in out:
        fd_cfg = cfg.get("model", {}).get("fd_regression", {})
        if bool(fd_cfg.get("enabled", False)):
            from .fractal import dbc_fractal_dimension
            # Compute FD target from GT mask (batch-level)
            fd_targets = []
            for b in range(targets.shape[0]):
                mask_patch = targets[b, 0]
                fd_val = dbc_fractal_dimension(mask_patch, box_sizes=(2, 4, 8, 16))
                fd_targets.append(fd_val)
            fd_target = torch.tensor(fd_targets, device=logits.device, dtype=torch.float32).unsqueeze(1)
            fd_w = float(fd_cfg.get("weight", 0.1))
            aux_loss = aux_loss + fd_w * F.mse_loss(out["fd_pred"], fd_target)

    return logits, aux_loss if aux_loss.item() > 0 else None


def train(
    cfg: dict[str, Any],
    use_synth_data: bool,
    overfit_one_batch: bool,
    run_dir: Path,
    require_gpu: bool = False,
    resume_from: str | None = None,
) -> None:
    seed = int(cfg.get("seed", 42))
    set_deterministic_seed(seed)
    device = resolve_device(require_gpu=require_gpu)

    train_cfg = cfg.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 2))
    accum_steps = int(train_cfg.get("accumulate_grad_steps", 1))
    steps_per_epoch = int(train_cfg.get("steps_per_epoch", 2))
    epochs = int(train_cfg.get("epochs", 1))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 0.0))
    save_every_epochs = int(train_cfg.get("save_every_epochs", 0))
    full_eval_every_epochs = int(train_cfg.get("full_eval_every_epochs", 0))
    full_eval_max_samples = int(train_cfg.get("full_eval_max_samples", 0))
    checkpoint_path = run_dir / "checkpoint.pt"

    # --- LR scheduler config (disabled by default) ---
    sched_cfg = train_cfg.get("scheduler", {})
    sched_type = str(sched_cfg.get("type", "none")).lower()
    sched_eta_min = float(sched_cfg.get("eta_min", 1e-6))

    # --- Mixed precision config (disabled by default) ---
    amp_enabled = bool(train_cfg.get("amp", False))
    use_cuda = device.type == "cuda"

    model = _build_model(cfg).to(device)
    if weight_decay > 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- LR scheduler ---
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    if sched_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1), eta_min=sched_eta_min,
        )
    elif sched_type == "step":
        step_size = int(sched_cfg.get("step_size", 30))
        gamma = float(sched_cfg.get("gamma", 0.1))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # --- AMP GradScaler ---
    scaler: torch.amp.GradScaler | None = None
    if amp_enabled and use_cuda:
        scaler = torch.amp.GradScaler()

    # --- Resume from checkpoint ---
    start_epoch = 0
    best_full_eval_dice = float("-inf")
    best_full_eval_cldice = 0.0
    best_full_eval_epoch = -1
    best_tau = 0.5
    if resume_from is not None:
        ckpt_path = Path(resume_from)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        print(f"Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt and scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and scaler is not None:
            scaler.load_state_dict(ckpt["scaler"])
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"])
            print(f"Resuming from epoch {start_epoch}/{epochs}")
        if "best_dice" in ckpt:
            best_full_eval_dice = float(ckpt["best_dice"])
            best_full_eval_epoch = int(ckpt.get("best_epoch", start_epoch))
            best_tau = float(ckpt.get("tau_star", 0.5))
            print(f"Best dice so far: {best_full_eval_dice:.4f} (epoch {best_full_eval_epoch}, tau*={best_tau:.2f})")
        del ckpt
        torch.cuda.empty_cache() if use_cuda else None

    lfd_params = LFDParams()
    loss_fn = CompositeLoss(cfg.get("loss", {})).to(device)

    samples = _prepare_manifest(cfg, use_synth_data=use_synth_data)
    write_manifest_and_hash(run_dir, samples)

    data_cfg = cfg.get("data", {})
    dataset_name = data_cfg.get("dataset")

    train_samples = [s for s in samples if s.get("split") == "train"] or samples
    if use_synth_data:
        batch = _make_synth_batch(train_samples, cfg, seed, batch_size)
        loader = [batch] if overfit_one_batch else [batch] * steps_per_epoch
        dataset = None
        train_indices: List[int] = []
    elif dataset_name == "hrf":
        dataset = build_dataset(cfg)
        # CRITICAL: inject split-assigned samples so train/val filtering works.
        dataset.manifest = samples
        train_indices = [i for i, s in enumerate(dataset.manifest) if s.get("split") == "train"]
        if not train_indices:
            print(f"WARNING: No train samples found in manifest splits: {set(s.get('split') for s in dataset.manifest)}")
            train_indices = list(range(len(dataset)))
        loader = None
    else:
        raise ValueError("Only synthetic data or HRF dataset is supported in this scaffold.")

    model.train()
    final_train_dice = None
    sampling_config = data_cfg.get("patch_sampling", {})
    fractal_cfg = cfg.get("fractal_prior", {})
    return_prior = bool(data_cfg.get("return_fractal_prior", False))
    photometric_config = data_cfg.get("photometric_aug", {})
    overfit_batch = None
    nca_freeze_until = int(train_cfg.get("nca_freeze_until_epoch", 0))
    # L1: Keep patch dataset as a local variable, not on model object
    patch_dataset = None
    latest_checkpoint_path = run_dir / "checkpoint_latest.pt"
    for epoch in range(start_epoch, epochs):
        # --- Optional NCA Curriculum Freeze ---
        if nca_freeze_until > 0 and hasattr(model, "nca_refiner") and model.nca_refiner is not None:
            if epoch == 0 and nca_freeze_until > 0:
                print(f"Phase A: NCA refiner frozen until epoch {nca_freeze_until}")
                for param in model.nca_refiner.parameters():
                    param.requires_grad = False
            elif epoch == nca_freeze_until:
                print(f"Phase B: Unfreezing NCA refiner at epoch {epoch}")
                for param in model.nca_refiner.parameters():
                    param.requires_grad = True

        if use_synth_data:
            epoch_loader = loader
        else:
            if patch_dataset is None:
                from .data import EpochPatchDataset
                patch_dataset = EpochPatchDataset(
                    base_dataset=dataset,
                    indices=train_indices,
                    length=steps_per_epoch * batch_size,
                    patch_size=data_cfg.get("patch_size", 96),
                    seed=seed,
                    sampling_config=sampling_config,
                    fractal_prior_config=fractal_cfg,
                    return_fractal_prior=return_prior,
                    photometric_config=photometric_config,
                )

            # Update epoch state so we get fresh patches next time
            patch_dataset.set_epoch(epoch)
            
            from torch.utils.data import DataLoader
            epoch_loader_obj = DataLoader(
                patch_dataset,
                batch_size=batch_size,
                shuffle=False,  # random state is deterministic based on epoch + index!
                num_workers=0,  # Keeps RAM low on 8GB machines
                pin_memory=(device.type == "cuda"),
                drop_last=True,
            )
            epoch_loader = epoch_loader_obj

        optimizer.zero_grad()
        for step, batch in enumerate(epoch_loader):
            x = batch["image"].to(device)
            y = batch["mask"].to(device)

            # --- Forward pass (optionally with AMP) ---
            if scaler is not None:
                with torch.amp.autocast(device_type="cuda"):
                    out, fractal_w = _forward_logits(model, x, data_cfg, lfd_params, batch=batch)
                    logits, aux_loss = _unpack_model_output(out, y, model, cfg)
                    loss, breakdown = loss_fn(logits, y, fractal_weight=fractal_w)
                    if aux_loss is not None:
                        loss = loss + aux_loss
            else:
                out, fractal_w = _forward_logits(model, x, data_cfg, lfd_params, batch=batch)
                logits, aux_loss = _unpack_model_output(out, y, model, cfg)
                loss, breakdown = loss_fn(logits, y, fractal_weight=fractal_w)
                if aux_loss is not None:
                    loss = loss + aux_loss

            loss = loss / accum_steps
            probs = torch.sigmoid(logits.float())  # always fp32 for metrics
            score = dice_score(probs, y)

            # --- Backward pass with optional AMP scaling + grad clipping ---
            if scaler is not None:
                scaler.scale(loss).backward()
                if (step + 1) % accum_steps == 0 or (step + 1) == len(epoch_loader):
                    if grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    # PDE-specific tighter clipping (log/sigmoid params get large grads)
                    _pde = getattr(model, 'fractal_diffusion', None)
                    if _pde is not None and hasattr(_pde, 'clip_pde_gradients'):
                        _pde.clip_pde_gradients(max_norm=0.5)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (step + 1) % accum_steps == 0 or (step + 1) == len(epoch_loader):
                    if grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    _pde = getattr(model, 'fractal_diffusion', None)
                    if _pde is not None and hasattr(_pde, 'clip_pde_gradients'):
                        _pde.clip_pde_gradients(max_norm=0.5)
                    optimizer.step()
                    optimizer.zero_grad()

            final_train_dice = float(score.item())
            if step == 0:
                cur_lr = optimizer.param_groups[0]["lr"]
                loss_keys = ", ".join([f"{k}={v:.4f}" for k, v in breakdown.items()])
                print(f"Epoch {epoch} step {step} loss {loss.item() * accum_steps:.4f} dice {score.item():.4f} lr {cur_lr:.2e} {loss_keys}")

        # --- Step LR scheduler at end of epoch ---
        if scheduler is not None:
            scheduler.step()

        # --- Log PDE parameters at end of epoch ---
        _pde_mod = getattr(model, 'fractal_diffusion', None)
        if _pde_mod is not None and hasattr(_pde_mod, 'get_param_dict'):
            import json as _json
            pde_log_path = run_dir / "pde_params.jsonl"
            pde_entry = {"epoch": epoch + 1, **_pde_mod.get_param_dict()}
            with open(pde_log_path, "a") as _f:
                _f.write(_json.dumps(pde_entry) + "\n")
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print(f"  PDE params: {', '.join(f'{k}={v:.4f}' for k, v in pde_entry.items() if k != 'epoch')}")

        # --- C2: Save checkpoint_latest.pt EVERY epoch for crash recovery ---
        _latest_ckpt: Dict[str, Any] = {
            "model": model.state_dict(),
            "config": cfg,
            "epoch": epoch + 1,
            "optimizer": optimizer.state_dict(),
        }
        if scheduler is not None:
            _latest_ckpt["scheduler"] = scheduler.state_dict()
        if scaler is not None:
            _latest_ckpt["scaler"] = scaler.state_dict()
        if best_full_eval_dice > float("-inf"):
            _latest_ckpt["best_dice"] = best_full_eval_dice
            _latest_ckpt["best_epoch"] = best_full_eval_epoch
            _latest_ckpt["tau_star"] = best_tau
        torch.save(_latest_ckpt, latest_checkpoint_path)

        if save_every_epochs > 0 and (epoch + 1) % save_every_epochs == 0:
            epoch_ckpt = run_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt"
            torch.save(_latest_ckpt, epoch_ckpt)

        if (
            full_eval_every_epochs > 0
            and dataset_name == "hrf"
            and (epoch + 1) % full_eval_every_epochs == 0
        ):
            # --- OOM-safe tiled eval: catch VRAM spikes, warn, continue ---
            try:
                model.eval()
                val_indices = [i for i, s in enumerate(dataset.manifest) if s.get("split") == "val"]
                if not val_indices:
                    val_indices = [i for i, s in enumerate(dataset.manifest) if s.get("split") == "train"]
                if not val_indices:
                    val_indices = list(range(len(dataset)))
                if full_eval_max_samples > 0:
                    val_indices = val_indices[:full_eval_max_samples]
                probs_list = []
                gts_list = []
                fov_list = []
                tiled_cfg = cfg.get("infer", {}).get("tiled", {})
                ph, pw = tiled_cfg.get("patch_size", [96, 96])
                sh, sw = tiled_cfg.get("stride", [48, 48])
                with torch.no_grad():
                    for idx in val_indices:
                        sample = dataset[idx]
                        image = sample["image"]
                        gt = sample["mask"]
                        probs = tiled_predict_proba(
                            model,
                            image,
                            patch_size=(int(ph), int(pw)),
                            stride=(int(sh), int(sw)),
                            blend=str(tiled_cfg.get("blend", "hann")),
                            pad_mode=str(tiled_cfg.get("pad_mode", "reflect")),
                            batch_tiles=int(tiled_cfg.get("batch_tiles", 8)),
                            device=device,
                            eps=float(tiled_cfg.get("eps", 1e-6)),
                            fractal_prior_config=fractal_cfg,
                        )
                        probs_list.append(probs.cpu())
                        gts_list.append(gt.unsqueeze(0).cpu())
                        if cfg.get("eval", {}).get("fov", {}).get("enabled", False) and "fov" in sample:
                            fov_list.append(sample["fov"].unsqueeze(0).unsqueeze(0).cpu().float())
                        else:
                            fov_list.append(torch.ones_like(gt.unsqueeze(0)).cpu())
                val_probs = torch.cat(probs_list, dim=0)
                y_val = torch.cat(gts_list, dim=0)
                fov_mask = torch.cat(fov_list, dim=0)
                val_probs = val_probs * fov_mask
                y_val = y_val * fov_mask
                sweep = sweep_thresholds(val_probs, y_val, _get_threshold_grid(cfg))
                epoch_best_dice = float(sweep["best_dice"])
                epoch_cldice = float(sweep.get("best_cldice", 0.0))
                if epoch_best_dice > best_full_eval_dice:
                    best_full_eval_dice = epoch_best_dice
                    best_full_eval_cldice = epoch_cldice
                    best_full_eval_epoch = epoch + 1
                    best_tau = float(sweep["tau_star"])
                    best_ckpt_path = run_dir / "checkpoint_best.pt"
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "config": cfg,
                            "epoch": epoch + 1,
                            "best_dice": best_full_eval_dice,
                            "best_cldice": best_full_eval_cldice,
                            "tau_star": best_tau,
                        },
                        best_ckpt_path,
                    )
                    print(
                        f"NEW_BEST_FULL_EVAL epoch={epoch + 1} best_dice={best_full_eval_dice:.4f} "
                        f"best_cldice={best_full_eval_cldice:.4f} tau*={best_tau:.2f} ckpt={best_ckpt_path}"
                    )
            except torch.cuda.OutOfMemoryError:
                print(
                    f"WARNING: OOM during full-image eval at epoch {epoch + 1}. "
                    f"Skipping this eval checkpoint — training continues."
                )
                torch.cuda.empty_cache()
            finally:
                # Explicit cleanup to release eval tensors
                probs_list = None  # type: ignore[assignment]
                gts_list = None  # type: ignore[assignment]
                fov_list = None  # type: ignore[assignment]
                import gc
                gc.collect()
                if use_cuda:
                    torch.cuda.empty_cache()
            model.train()

    model.eval()
    with torch.no_grad():
        val_samples = [s for s in samples if s.get("split") == "val"] or train_samples
        if use_synth_data:
            val_batch = _make_synth_batch(val_samples, cfg, seed + 1, batch_size=1)
        elif dataset_name == "hrf":
            dataset = build_dataset(cfg)
            # CRITICAL: inject split-assigned samples for proper val filtering.
            dataset.manifest = samples
            val_indices = [i for i, s in enumerate(dataset.manifest) if s.get("split") == "val"]
            if not val_indices:
                val_indices = [i for i, s in enumerate(dataset.manifest) if s.get("split") == "train"]
            if not val_indices:
                val_indices = list(range(len(dataset)))

            val_sample = dataset[val_indices[0]]
            val_patch_ds = InfiniteRandomPatchDataset(
                image=val_sample["image"],
                mask=val_sample["mask"],
                patch_size=data_cfg.get("patch_size", 96),
                seed=seed + 1,
                sampling_config=data_cfg.get("patch_sampling", {}),
                sample_id=str(val_sample.get("id", val_indices[0])),
                fractal_prior_config=cfg.get("fractal_prior", {}),
                return_fractal_prior=bool(data_cfg.get("return_fractal_prior", False)),
                photometric_config={},
            )
            val_patch = next(iter(val_patch_ds))
            val_batch = {
                "image": val_patch["image"].unsqueeze(0),
                "mask": val_patch["mask"].unsqueeze(0),
            }
            if bool(data_cfg.get("return_fractal_prior", False)) and "prior" in val_patch:
                val_batch["prior"] = val_patch["prior"].unsqueeze(0)
        else:
            raise ValueError("Only synthetic data or HRF dataset is supported in this scaffold.")

        x_val = val_batch["image"].to(device)
        y_val = val_batch["mask"].to(device)

        val_out, _ = _forward_logits(model, x_val, data_cfg, lfd_params, batch=val_batch)
        val_logits = val_out["logits"] if isinstance(val_out, dict) else val_out

        val_score = dice_score(torch.sigmoid(val_logits), y_val)
        final_val_dice = float(val_score.item())

    # C3: Save full resume state in final checkpoint (not just model + config)
    final_ckpt: Dict[str, Any] = {
        "model": model.state_dict(),
        "config": cfg,
        "epoch": epochs,
        "optimizer": optimizer.state_dict(),
    }
    if scheduler is not None:
        final_ckpt["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        final_ckpt["scaler"] = scaler.state_dict()
    if best_full_eval_dice > float("-inf"):
        final_ckpt["best_dice"] = best_full_eval_dice
        final_ckpt["best_epoch"] = best_full_eval_epoch
        final_ckpt["tau_star"] = best_tau
    torch.save(final_ckpt, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"FINAL_METRICS train_dice={final_train_dice:.4f} val_dice={final_val_dice:.4f}")

    metrics = {
        "train_dice": final_train_dice,
        "val_dice": final_val_dice,
        "checkpoint_path": str(checkpoint_path),
    }
    if best_full_eval_epoch > 0:
        metrics["best_full_eval_dice"] = best_full_eval_dice
        metrics["best_full_eval_cldice"] = best_full_eval_cldice
        metrics["best_full_eval_epoch"] = best_full_eval_epoch
        metrics["best_full_eval_tau_star"] = best_tau
        metrics["best_checkpoint_path"] = str(run_dir / "checkpoint_best.pt")
    metrics.update(breakdown)
    write_metrics(run_dir, metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Fractal-Prior Swin-UNet.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--use_synth_data", action="store_true", help="Use synthetic data.")
    parser.add_argument("--overfit_one_batch", action="store_true", help="Overfit on one batch.")
    parser.add_argument("--run_id", type=str, default=None, help="Run identifier.")
    parser.add_argument("--run_base", type=str, default="runs", help="Base directory for runs.")
    parser.add_argument(
        "--require_gpu",
        action="store_true",
        help="Fail if CUDA is not available (do not fall back to CPU).",
    )
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume training from.")
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

    train(
        cfg,
        use_synth_data=args.use_synth_data,
        overfit_one_batch=args.overfit_one_batch,
        run_dir=run_dir,
        require_gpu=args.require_gpu,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
