#!/usr/bin/env python3
"""Generate clean ablation configs that isolate EXACTLY one variable per step.

The reviewer-proof ablation matrix:

   Config   |  Prior                | Injection        | Topo Loss         | FreqGeom
  ----------|--------------------  -|------------------|-------------------|---------
   A_base   |  None                 | —                | Dice+BCE+Focal+cl | No
   B_gate   |  LFD (1ch)            | Sigmoid gate     | Dice+BCE+Focal+cl | No
   C_spade1 |  LFD (1ch)            | SPADE v1         | Dice+BCE+Focal+cl | No
   D_spade2 |  LFD+PDE (3ch)        | SPADE v2         | Dice+BCE+Focal+cl | No
   E_hess   |  LFD+PDE+Hess (5ch)   | SPADE v2         | Dice+BCE+Focal+cl | No
   F_skel   |  LFD+PDE+Hess (5ch)   | SPADE v2         | +SkelRecall       | No
   G_fbce   |  LFD+PDE+Hess (5ch)   | SPADE v2         | +SkelRecall+FrBCE | No
   H_freq   |  LFD+PDE+Hess (5ch)   | SPADE v2         | +SkelRecall+FrBCE | Yes

Each row differs from the previous by EXACTLY one change.
"""

import copy
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent / "configs" / "ablation_v2"
BASE_DIR.mkdir(exist_ok=True)

# ── Shared sections ──────────────────────────────────────────────────────

SHARED_TRAIN = {
    "epochs": 60,
    "steps_per_epoch": 220,
    "batch_size": 1,
    "lr": 0.0005,
    "weight_decay": 0.01,
    "amp": True,
    "grad_clip_norm": 1.0,
    "scheduler": {"type": "cosine", "eta_min": 1e-6},
    "save_every_epochs": 10,
    "full_eval_every_epochs": 10,
    "full_eval_max_samples": 0,
}

SHARED_DATA = {
    "dataset": "hrf",
    "dataset_root": "/home/lucian/medical-imaging-ai/Retina Datasets/Full retina dataset/HRF",
    "manifest_path": "manifests/hrf_full.jsonl",
    "split": {"mode": "sample", "seed": 123, "ratios": {"train": 0.6, "val": 0.2, "test": 0.2}},
    "patch_size": [384, 384],
    "patch_sampling": {
        "enabled": True, "mode": "vessel_aware",
        "p_vessel": 0.7, "p_background": 0.3,
        "vessel_buffer": 3, "background_buffer": 2,
    },
    "return_fov": True,
    "photometric_aug": {"enabled": True, "brightness": 0.08, "contrast": 0.15, "gamma": 0.15},
    "return_fractal_prior": True,
}

SHARED_EVAL = {
    "full_image": True,
    "threshold": {"default": 0.5, "use_tau_star_if_available": True},
    "threshold_sweep": {
        "enabled": True,
        "grid": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
    },
    "fov": {"enabled": True, "key": "fov_mask_path", "ignore_outside": True},
}

SHARED_INFER = {
    "mode": "tiled",
    "tiled": {
        "patch_size": [384, 384], "stride": [192, 192],
        "pad_mode": "reflect", "blend": "hann", "batch_tiles": 4, "eps": 1e-6,
    },
}

# ── Config Definitions ───────────────────────────────────────────────────

CONFIGS = {}

# A: Pure baseline — no priors, standard losses
CONFIGS["A_baseline"] = {
    "comment": "Pure Swin-UNet baseline. No fractal priors. Standard losses only.",
    "model": {
        "name": "fractal_prior", "in_channels": 3, "embed_dim": 32,
        "depths": [1, 1, 1], "enable_fractal_gate": False, "alpha": 1.0,
        "prior_fusion": {"enabled": False}, "prior_channels": 1,
        "fractal_spade": {"enabled": False},
        "fractal_diffusion_spade": {"enabled": False},
        "freq_geom": {"enabled": False},
    },
    "loss": {
        "focal": {"enabled": True, "weight": 0.1},
        "cldice": {"enabled": True, "weight": 0.12, "iter": 10},
        "skeleton_recall": {"enabled": False},
        "fractal_bce": {"enabled": False},
    },
}

# B: LFD via sigmoid gate (naive injection)
# CHANGE from A: + LFD sigmoid gate
CONFIGS["B_lfd_gate"] = copy.deepcopy(CONFIGS["A_baseline"])
CONFIGS["B_lfd_gate"]["comment"] = "LFD prior via sigmoid gate. Tests naive prior injection."
CONFIGS["B_lfd_gate"]["model"]["enable_fractal_gate"] = True

# C: LFD via SPADE v1 (injection mechanism change)
# CHANGE from B: gate → SPADE v1 (same 1ch LFD prior)
CONFIGS["C_lfd_spade1"] = copy.deepcopy(CONFIGS["A_baseline"])
CONFIGS["C_lfd_spade1"]["comment"] = "LFD prior via SPADE v1. Tests injection mechanism."
CONFIGS["C_lfd_spade1"]["model"]["fractal_spade"] = {"enabled": True, "hidden_dim": 32}

# D: LFD+PDE via SPADE v2 (prior content change)
# CHANGE from C: SPADE v1(1ch) → SPADE v2(3ch LFD+PDE+edge)
CONFIGS["D_pde_spade2"] = copy.deepcopy(CONFIGS["A_baseline"])
CONFIGS["D_pde_spade2"]["comment"] = "LFD+PDE+edge (3ch) via SPADE v2. Tests learnable PDE value."
CONFIGS["D_pde_spade2"]["model"]["fractal_diffusion_spade"] = {
    "enabled": True, "n_steps": 5, "prior_channels": 3, "hidden_dim": 32,
    "use_hessian_direction": False,
}

# E: LFD+PDE+Hessian via SPADE v2 (add direction field)
# CHANGE from D: 3ch → 5ch (add Hessian cos θ, sin θ)
CONFIGS["E_hessian"] = copy.deepcopy(CONFIGS["D_pde_spade2"])
CONFIGS["E_hessian"]["comment"] = "5ch prior (+Hessian direction). Tests direction field value."
CONFIGS["E_hessian"]["model"]["fractal_diffusion_spade"]["prior_channels"] = 5
CONFIGS["E_hessian"]["model"]["fractal_diffusion_spade"]["use_hessian_direction"] = True

# F: Add skeleton recall loss
# CHANGE from E: + skeleton recall loss
CONFIGS["F_skel_recall"] = copy.deepcopy(CONFIGS["E_hessian"])
CONFIGS["F_skel_recall"]["comment"] = "5ch SPADE v2 + skeleton recall loss. Tests topo loss value."
CONFIGS["F_skel_recall"]["loss"] = copy.deepcopy(CONFIGS["E_hessian"]["loss"])
CONFIGS["F_skel_recall"]["loss"]["skeleton_recall"] = {"enabled": True, "weight": 0.15, "iter": 10}

# G: Add fractal-weighted BCE
# CHANGE from F: + fractal-weighted BCE
CONFIGS["G_fractal_bce"] = copy.deepcopy(CONFIGS["F_skel_recall"])
CONFIGS["G_fractal_bce"]["comment"] = "5ch SPADE v2 + SkelRecall + FracBCE. Tests FD-weighted loss."
CONFIGS["G_fractal_bce"]["loss"]["fractal_bce"] = {"enabled": True, "weight": 0.1, "alpha": 1.5}

# H: Add FreqGeom decomposition
# CHANGE from G: + FreqGeom decoder
CONFIGS["H_full"] = copy.deepcopy(CONFIGS["G_fractal_bce"])
CONFIGS["H_full"]["comment"] = "Full stack: 5ch SPADE v2 + all losses + FreqGeom. Tests decoder value."
CONFIGS["H_full"]["model"]["freq_geom"] = {"enabled": True}

# ── Write configs ────────────────────────────────────────────────────────
for name, cfg in CONFIGS.items():
    comment = cfg.pop("comment", "")

    full_cfg = {
        "seed": 42,
        "model": cfg["model"],
        "train": copy.deepcopy(SHARED_TRAIN),
        "data": copy.deepcopy(SHARED_DATA),
        "eval": copy.deepcopy(SHARED_EVAL),
        "infer": copy.deepcopy(SHARED_INFER),
        "loss": cfg["loss"],
        "fractal_prior": {"enabled": True},
    }

    path = BASE_DIR / f"{name}.yaml"
    with open(path, "w") as f:
        f.write(f"# {name}: {comment}\n")
        yaml.dump(full_cfg, f, default_flow_style=False, sort_keys=False)

    print(f"  ✓ {path.name}")

print(f"\nGenerated {len(CONFIGS)} configs in {BASE_DIR}")
print("\nAblation matrix:")
print("  A_baseline    → Pure Swin-UNet (no priors)")
print("  B_lfd_gate    → + LFD via sigmoid gate       [Δ: +prior injection]")
print("  C_lfd_spade1  → + LFD via SPADE v1            [Δ: injection mechanism]")
print("  D_pde_spade2  → + PDE via SPADE v2 (3ch)      [Δ: prior content + SPADE v2]")
print("  E_hessian     → + Hessian direction (5ch)      [Δ: direction field]")
print("  F_skel_recall → + Skeleton recall loss         [Δ: topo loss]")
print("  G_fractal_bce → + Fractal-weighted BCE         [Δ: FD-weighted loss]")
print("  H_full        → + FreqGeom decoder             [Δ: decoder module]")
