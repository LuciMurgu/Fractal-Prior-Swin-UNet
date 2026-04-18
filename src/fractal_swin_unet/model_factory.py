"""Unified model factory — single source of truth for model construction.

This eliminates the dangerous _build_model copy-paste drift between
train.py, eval.py, and infer.py (C1 in code review).
"""

from __future__ import annotations

from typing import Any

import torch

from .data.wavelets import wavelet_channel_count
from .model import SimpleSwinUNet
from .models import FractalPriorSwinUNet, SwinUNetTiny


def resolve_in_channels(cfg: dict[str, Any]) -> int:
    """Compute the number of input channels after optional wavelet expansion."""
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    base = int(model_cfg.get("in_channels", 3))
    if bool(data_cfg.get("use_green_channel", False)):
        base = 1
    return wavelet_channel_count(base, data_cfg.get("wavelet", {}))


def build_model(cfg: dict[str, Any]) -> torch.nn.Module:
    """Build a model from a resolved config dict.

    Supports all model types and all feature flags (SPADE, rot-equivariant,
    hessian direction, NCA, multi-decoder, FD regression, freq-geom).
    """
    model_cfg = cfg.get("model", {})
    name = model_cfg.get("name", "simple")
    in_channels = resolve_in_channels(cfg)
    embed_dim = int(model_cfg.get("embed_dim", 32))
    depths = tuple(model_cfg.get("depths", [1, 1, 1]))
    drop_rate = float(model_cfg.get("drop_rate", 0.0))

    if name in ("baseline", "swin_unet"):
        return SwinUNetTiny(
            in_channels=in_channels,
            embed_dim=embed_dim,
            depths=depths,
            drop_rate=drop_rate,
        )

    if name == "fractal_prior":
        enable_gate = bool(model_cfg.get("enable_fractal_gate", True))
        prior_fusion_cfg = model_cfg.get("prior_fusion", {})
        enable_prior_fusion = bool(prior_fusion_cfg.get("enabled", False))
        alpha = float(model_cfg.get("alpha", 1.0))

        # NCA refinement head (V7) — disabled by default
        nca_cfg = model_cfg.get("nca", {})
        use_nca = bool(nca_cfg.get("enabled", False))
        nca_hidden = int(nca_cfg.get("hidden", 32))
        nca_steps = int(nca_cfg.get("steps", 16))
        nca_stochastic = float(nca_cfg.get("stochastic_rate", 0.5))

        # V9 features — multi-channel prior, multi-decoder, FD regression
        prior_channels = int(model_cfg.get("prior_channels", 1))
        multi_dec_cfg = model_cfg.get("multi_decoder", {})
        use_multi_decoder = bool(multi_dec_cfg.get("enabled", False))
        use_fd_regression = bool(
            model_cfg.get("fd_regression", {}).get("enabled", False)
        )
        use_freq_geom = bool(
            model_cfg.get("freq_geom", {}).get("enabled", False)
        )

        # Fractal SPADE (learnable prior modulation)
        spade_cfg = model_cfg.get("fractal_spade", {})
        use_fractal_spade = bool(spade_cfg.get("enabled", False))
        spade_hidden = int(spade_cfg.get("hidden_dim", 32))

        # Fractal Anisotropic Diffusion + SPADE v2 (advanced PDE conditioning)
        fad_cfg = model_cfg.get("fractal_diffusion_spade", {})
        use_fad = bool(fad_cfg.get("enabled", False))
        fad_n_steps = int(fad_cfg.get("n_steps", 5))
        fad_prior_channels = int(fad_cfg.get("prior_channels", 3))
        use_hessian_dir = bool(fad_cfg.get("use_hessian_direction", False))

        # Rotation-equivariant convolutions
        rot_cfg = model_cfg.get("rot_equivariant", {})
        use_rot_eq = bool(rot_cfg.get("enabled", False))
        n_rotations = int(rot_cfg.get("n_rotations", 8))

        return FractalPriorSwinUNet(
            in_channels=in_channels,
            embed_dim=embed_dim,
            depths=depths,
            enable_fractal_gate=enable_gate,
            enable_prior_fusion=enable_prior_fusion,
            alpha=alpha,
            drop_rate=drop_rate,
            use_nca_refiner=use_nca,
            nca_hidden=nca_hidden,
            nca_steps=nca_steps,
            nca_stochastic_rate=nca_stochastic,
            prior_channels=prior_channels,
            use_multi_decoder=use_multi_decoder,
            use_fd_regression=use_fd_regression,
            use_freq_geom=use_freq_geom,
            use_fractal_spade=use_fractal_spade,
            spade_hidden=spade_hidden,
            use_fractal_diffusion_spade=use_fad,
            fad_n_steps=fad_n_steps,
            fad_prior_channels=fad_prior_channels,
            use_hessian_direction=use_hessian_dir,
            use_rot_equivariant=use_rot_eq,
            n_rotations=n_rotations,
        )

    return SimpleSwinUNet(in_channels=in_channels)
