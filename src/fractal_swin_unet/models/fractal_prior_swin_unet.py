"""Fractal-Prior Swin-UNet with LFD gating and learnable PDE conditioning."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..fractal import LFDParams, compute_lfd_map
from .gating import FractalSPADE, FractalSPADEv2, apply_fractal_gate
from .swin_unet import SwinBlock, SwinStage


def _batch_lfd_map(x: torch.Tensor, params: LFDParams) -> torch.Tensor:
    lfd_maps = []
    for sample in x:
        lfd = compute_lfd_map(sample, params=params)
        lfd_maps.append(lfd)
    return torch.stack(lfd_maps, dim=0).unsqueeze(1)


class FractalPriorSwinUNet(nn.Module):
    """Swin-UNet with fractal gating, optional multi-decoder, and FD regression."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 32,
        depths: tuple[int, int, int] = (1, 1, 1),
        enable_fractal_gate: bool = True,
        enable_prior_fusion: bool = False,
        alpha: float = 1.0,
        lfd_params: Optional[LFDParams] = None,
        drop_rate: float = 0.0,
        use_nca_refiner: bool = False,
        nca_hidden: int = 32,
        nca_steps: int = 16,
        nca_stochastic_rate: float = 0.5,
        prior_channels: int = 1,
        use_multi_decoder: bool = False,
        use_fd_regression: bool = False,
        use_freq_geom: bool = False,
        use_fractal_spade: bool = False,
        spade_hidden: int = 32,
        use_fractal_diffusion_spade: bool = False,
        fad_n_steps: int = 5,
        fad_prior_channels: int = 3,
        use_hessian_direction: bool = False,
        use_rot_equivariant: bool = False,
        n_rotations: int = 8,
    ) -> None:
        super().__init__()
        self.enable_fractal_gate = enable_fractal_gate
        self.enable_prior_fusion = enable_prior_fusion
        self.use_fractal_spade = use_fractal_spade
        self.alpha = alpha
        self.lfd_params = lfd_params or LFDParams()
        self.use_nca_refiner = use_nca_refiner
        self.prior_channels = prior_channels
        self.use_multi_decoder = use_multi_decoder
        self.use_fd_regression = use_fd_regression
        self.use_freq_geom = use_freq_geom
        self.use_fractal_diffusion_spade = use_fractal_diffusion_spade
        self.use_hessian_direction = use_hessian_direction
        self.use_rot_equivariant = use_rot_equivariant

        self.stem = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)

        # Choose standard or rotation-equivariant blocks
        if use_rot_equivariant:
            from .rot_equivariant import RotEquivariantSwinBlock, RotEquivariantSwinStage
            _Stage = lambda ic, oc, d, ds, dr: RotEquivariantSwinStage(
                ic, oc, d, ds, n_rotations=n_rotations, drop_rate=dr,
            )
            _Block = lambda c, dr: RotEquivariantSwinBlock(
                c, n_rotations=n_rotations, drop_rate=dr,
            )
        else:
            _Stage = lambda ic, oc, d, ds, dr: SwinStage(ic, oc, d, ds, drop_rate=dr)
            _Block = lambda c, dr: SwinBlock(c, drop_rate=dr)

        self.enc1 = _Stage(embed_dim, embed_dim, depths[0], False, drop_rate)
        self.enc2 = _Stage(embed_dim, embed_dim * 2, depths[1], True, drop_rate)
        self.enc3 = _Stage(embed_dim * 2, embed_dim * 4, depths[2], True, drop_rate)

        self.bottleneck = _Block(embed_dim * 4, drop_rate)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(embed_dim * 4 + embed_dim * 2, embed_dim * 2, kernel_size=3, padding=1),
            _Block(embed_dim * 2, drop_rate),
        )

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(embed_dim * 2 + embed_dim, embed_dim, kernel_size=3, padding=1),
            _Block(embed_dim, drop_rate),
        )

        if self.enable_prior_fusion:
            self.prior_to_s1 = nn.Conv2d(prior_channels, embed_dim, kernel_size=1)
            self.prior_to_s2 = nn.Conv2d(prior_channels, embed_dim * 2, kernel_size=1)
            self.fuse_s1 = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1)
            self.fuse_s2 = nn.Conv2d(embed_dim * 4, embed_dim * 2, kernel_size=1)
        else:
            self.prior_to_s1 = None
            self.prior_to_s2 = None
            self.fuse_s1 = None
            self.fuse_s2 = None

        # --- Fractal SPADE modulation (replaces hand-crafted gate) ---
        if use_fractal_spade:
            self.spade_s1 = FractalSPADE(embed_dim, prior_channels=1, hidden_dim=spade_hidden)
            self.spade_s2 = FractalSPADE(embed_dim * 2, prior_channels=1, hidden_dim=spade_hidden)
        else:
            self.spade_s1 = None
            self.spade_s2 = None

        # --- Fractal Anisotropic Diffusion + SPADE v2 (advanced) ---
        self.fractal_diffusion = None
        self.spade_v2_s1 = None
        self.spade_v2_s2 = None
        if use_fractal_diffusion_spade:
            from ..pde.fractal_diffusion import FractalAnisotropicDiffusion
            self.fractal_diffusion = FractalAnisotropicDiffusion(n_steps=fad_n_steps)
            self.spade_v2_s1 = FractalSPADEv2(
                embed_dim, prior_channels=fad_prior_channels, hidden_dim=spade_hidden,
            )
            self.spade_v2_s2 = FractalSPADEv2(
                embed_dim * 2, prior_channels=fad_prior_channels, hidden_dim=spade_hidden,
            )

        self.out_conv = nn.Conv2d(embed_dim, 1, kernel_size=1)

        # --- Optional NCA refinement head ---
        self.nca_refiner = None
        if use_nca_refiner:
            from .fractal_nca import FractalNCA
            self.nca_refiner = FractalNCA(
                c_logit=1,
                c_skip=embed_dim,
                c_hidden=nca_hidden,
                n_steps=nca_steps,
                stochastic_rate=nca_stochastic_rate,
            )

        # --- Optional multi-decoder (edge + skeleton) ---
        self.edge_decoder = None
        self.skeleton_decoder = None
        if use_multi_decoder:
            from .multi_decoder import EdgeDecoder, SkeletonDecoder
            self.edge_decoder = EdgeDecoder(embed_dim, drop_rate=drop_rate)
            self.skeleton_decoder = SkeletonDecoder(embed_dim, drop_rate=drop_rate)

        # --- Optional FD regression head ---
        self.fd_head = None
        if use_fd_regression:
            from .multi_decoder import FDRegressionHead
            self.fd_head = FDRegressionHead(embed_dim * 4)

        # --- Optional Frequency-Geometric Decomposition (FGOS-Net 2026) ---
        self.freq_geom_dec2 = None
        self.freq_geom_dec1 = None
        if use_freq_geom:
            from .freq_geom import FreqGeomDecomposition
            self.freq_geom_dec2 = FreqGeomDecomposition(embed_dim * 2)
            self.freq_geom_dec1 = FreqGeomDecomposition(embed_dim)

    def _downsample_lfd(self, lfd_map: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(lfd_map, size=size, mode="bilinear", align_corners=False)

    def _aggregate_prior(self, lfd_map: torch.Tensor) -> torch.Tensor:
        if lfd_map.ndim != 4:
            raise ValueError("lfd_map must be 4D tensor.")
        if self.prior_channels == 1:
            return lfd_map.mean(dim=1, keepdim=True)
        return lfd_map

    def forward(
        self,
        x: torch.Tensor,
        lfd_map: Optional[torch.Tensor] = None,
        expect_prior: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        needs_prior = (
            self.enable_fractal_gate or self.enable_prior_fusion
            or self.use_fractal_spade or self.use_fractal_diffusion_spade
        )
        if needs_prior:
            if lfd_map is None:
                if expect_prior:
                    raise ValueError("Expected prior map but none was provided.")
                lfd_map = _batch_lfd_map(x, self.lfd_params)
            if lfd_map.ndim != 4:
                raise ValueError("lfd_map must have shape (B, C, H, W).")
        else:
            lfd_map = None

        # Store grayscale input for fractal diffusion PDE
        if self.use_fractal_diffusion_spade:
            if x.shape[1] == 3:
                self._input_gray = (
                    0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
                ).detach()
            else:
                self._input_gray = x[:, 0:1].detach()
            # Normalize to [0, 1]
            ig_min = self._input_gray.amin(dim=(-2, -1), keepdim=True)
            ig_max = self._input_gray.amax(dim=(-2, -1), keepdim=True)
            self._input_gray = (self._input_gray - ig_min) / (ig_max - ig_min + 1e-8)

        x = self.stem(x)
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)

        bottleneck_feats = self.bottleneck(s3)

        if lfd_map is not None:
            lfd_s2 = self._downsample_lfd(lfd_map, s2.shape[2:])
            lfd_s1 = self._downsample_lfd(lfd_map, s1.shape[2:])
            # For gating, always reduce to 1 channel
            lfd_s2_gate = lfd_s2.mean(dim=1, keepdim=True)
            lfd_s1_gate = lfd_s1.mean(dim=1, keepdim=True)

            if (
                self.use_fractal_diffusion_spade
                and self.fractal_diffusion is not None
                and self.spade_v2_s1 is not None
                and self.spade_v2_s2 is not None
            ):
                # --- Fractal Anisotropic Diffusion + SPADE v2 ---
                # Convert input image to grayscale for the PDE
                x_input = self.stem.weight.new_zeros(1)  # get device
                # Use the original lfd_map (1-ch) and original image (from x before stem)
                # We stored the pre-stem input in the forward call
                gray = self._input_gray  # (B, 1, H, W) set earlier
                lfd_1ch = lfd_map.mean(dim=1, keepdim=True)  # (B, 1, H, W)

                # Run learnable PDE
                enhanced, edge_residual = self.fractal_diffusion(gray, lfd_1ch)

                # Build prior stack: [LFD, enhanced, edge_residual, (cos θ, sin θ)]
                prior_channels = [lfd_1ch, enhanced, edge_residual]

                if self.use_hessian_direction:
                    from ..pde.hessian_direction import hessian_direction_field
                    # Compute direction field from grayscale input
                    dir_field = hessian_direction_field(
                        gray, sigmas=(1.0, 2.0, 4.0)
                    )  # (B, 2, H, W)
                    prior_channels.append(dir_field)

                prior_stack = torch.cat(prior_channels, dim=1)

                # Downsample prior stack to each skip resolution
                prior_s2 = self._downsample_lfd(prior_stack, s2.shape[2:])
                prior_s1 = self._downsample_lfd(prior_stack, s1.shape[2:])

                # Apply SPADE v2 modulation
                s2 = self.spade_v2_s2(s2, prior_s2)
                s1 = self.spade_v2_s1(s1, prior_s1)

            elif self.use_fractal_spade and self.spade_s2 is not None and self.spade_s1 is not None:
                # SPADE modulation (learnable, replaces hand-crafted gate)
                s2 = self.spade_s2(s2, lfd_s2_gate)
                s1 = self.spade_s1(s1, lfd_s1_gate)
            elif self.enable_fractal_gate:
                s2 = apply_fractal_gate(s2, lfd_s2_gate, alpha=self.alpha)
                s1 = apply_fractal_gate(s1, lfd_s1_gate, alpha=self.alpha)

            if self.enable_prior_fusion:
                if self.prior_to_s1 is None or self.prior_to_s2 is None or self.fuse_s1 is None or self.fuse_s2 is None:
                    raise RuntimeError("Prior fusion is enabled but fusion layers are not initialized.")
                # For fusion, use full multi-channel prior
                lfd_s2_fuse = self._aggregate_prior(lfd_s2)
                lfd_s1_fuse = self._aggregate_prior(lfd_s1)
                p2 = self.prior_to_s2(lfd_s2_fuse)
                p1 = self.prior_to_s1(lfd_s1_fuse)
                s2 = self.fuse_s2(torch.cat([s2, p2], dim=1))
                s1 = self.fuse_s1(torch.cat([s1, p1], dim=1))

        x = self.up2(bottleneck_feats)
        x = torch.cat([x, s2], dim=1)
        x = self.dec2(x)
        if self.freq_geom_dec2 is not None:
            x = self.freq_geom_dec2(x)

        x = self.up1(x)
        x = torch.cat([x, s1], dim=1)
        x = self.dec1(x)
        if self.freq_geom_dec1 is not None:
            x = self.freq_geom_dec1(x)

        coarse_logits = self.out_conv(x)

        if self.nca_refiner is not None:
            refined_logits = self.nca_refiner(coarse_logits, s1)
            main_logits = refined_logits
        else:
            main_logits = coarse_logits

        # If multi-decoder or FD regression, return dict
        if self.use_multi_decoder or self.use_fd_regression:
            result: dict[str, torch.Tensor] = {"logits": main_logits}
            if self.edge_decoder is not None:
                result["edge_logits"] = self.edge_decoder(x)
            if self.skeleton_decoder is not None:
                result["skeleton_logits"] = self.skeleton_decoder(x)
            if self.fd_head is not None:
                result["fd_pred"] = self.fd_head(bottleneck_feats)
            return result

        return main_logits

