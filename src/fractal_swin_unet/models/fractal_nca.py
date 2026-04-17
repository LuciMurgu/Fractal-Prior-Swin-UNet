"""Neural Cellular Automata (NCA) refinement head for Fractal U-Net V7.

The NCA acts as a "Gardener" that iteratively refines the coarse segmentation
produced by the U-Net "Architect". It learns local growth/pruning rules that
repair fragmented vessels and enforce topological connectivity.

Architecture:
  1. State initialization: project [coarse_logits, fractal_feats, encoder_skip]
     into a hidden state of shape (B, C_hidden, H, W).
  2. Perception: fixed Sobel-x and Sobel-y kernels compute local gradients.
     Combined with identity → 3×C_hidden perception channels.
  3. Update rule: two 1×1 convolutions (MLP) with ReLU produce a state delta.
  4. Stochastic mask: Bernoulli mask (p=0.5) prevents grid artifacts.
  5. Iterative loop: state += mask * delta, repeated T times.
  6. Readout: 1×1 conv projects first channel to refined logits.

The entire loop uses gradient checkpointing (chunks of 4) to keep VRAM
usage bounded on the RTX 4070 (8GB).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


class FractalNCA(nn.Module):
    """Neural Cellular Automata post-decoder refinement head.

    Args:
        c_logit: Number of logit channels from the U-Net (typically 1).
        c_skip: Number of encoder skip channels at depth-0.
        c_hidden: Hidden state channels for the NCA.
        n_steps: Number of NCA iteration steps.
        stochastic_rate: Probability of updating each cell per step.
        checkpoint_every: Gradient checkpoint every N steps.
    """

    def __init__(
        self,
        c_logit: int = 1,
        c_skip: int = 32,
        c_hidden: int = 32,
        n_steps: int = 16,
        stochastic_rate: float = 0.5,
        checkpoint_every: int = 4,
    ) -> None:
        super().__init__()
        self.c_hidden = c_hidden
        self.n_steps = n_steps
        self.stochastic_rate = stochastic_rate
        self.checkpoint_every = checkpoint_every

        # --- State initialization: project inputs → hidden state ---
        c_in = c_logit + c_skip
        self.state_init = nn.Sequential(
            nn.Conv2d(c_in, c_hidden, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # --- Fixed Sobel perception kernels (not learned) ---
        # Registers: sobel_x, sobel_y as (1, 1, 3, 3) grouped conv kernels
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).reshape(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).reshape(1, 1, 3, 3) / 8.0
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        # --- Learned update rule (1×1 conv MLP) ---
        # Perception produces 3×C_hidden channels (identity + sobel_x + sobel_y)
        c_perceive = 3 * c_hidden
        self.update_rule = nn.Sequential(
            nn.Conv2d(c_perceive, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, c_hidden, kernel_size=1),
        )
        # Zero-init the last conv so the NCA starts as identity
        nn.init.zeros_(self.update_rule[-1].weight)
        nn.init.zeros_(self.update_rule[-1].bias)

        # --- Readout head: hidden → logits ---
        self.readout = nn.Conv2d(c_hidden, c_logit, kernel_size=1)

    def _perceive(self, state: torch.Tensor) -> torch.Tensor:
        """Apply fixed Sobel perception to each channel via grouped conv.

        Args:
            state: (B, C_hidden, H, W) current NCA state.

        Returns:
            Perception tensor of shape (B, 3*C_hidden, H, W).
        """
        b, c, h, w = state.shape

        # Expand Sobel kernels for grouped convolution: (C, 1, 3, 3)
        sx = self.sobel_x.expand(c, -1, -1, -1)
        sy = self.sobel_y.expand(c, -1, -1, -1)

        grad_x = F.conv2d(state, sx, padding=1, groups=c)
        grad_y = F.conv2d(state, sy, padding=1, groups=c)

        return torch.cat([state, grad_x, grad_y], dim=1)

    def _step(self, state: torch.Tensor) -> torch.Tensor:
        """Single NCA step: perceive → update → stochastic mask.

        Args:
            state: (B, C_hidden, H, W) current state.

        Returns:
            Updated state tensor of the same shape.
        """
        perception = self._perceive(state)
        delta = self.update_rule(perception)

        # Stochastic update mask (only during training)
        if self.training and self.stochastic_rate < 1.0:
            mask = (
                torch.rand(
                    state.shape[0], 1, state.shape[2], state.shape[3],
                    device=state.device, dtype=state.dtype,
                )
                < self.stochastic_rate
            ).float()
            delta = delta * mask

        return state + delta

    def _checkpointed_steps(self, state: torch.Tensor, n: int) -> torch.Tensor:
        """Run n steps with gradient checkpointing.

        Args:
            state: Current NCA state.
            n: Number of steps to run in this chunk.

        Returns:
            State after n steps.
        """
        for _ in range(n):
            state = self._step(state)
        return state

    def forward(
        self,
        coarse_logits: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        """Run NCA refinement.

        Args:
            coarse_logits: (B, 1, H, W) coarse logits from U-Net decoder.
            skip: (B, C_skip, H, W) encoder skip features at depth-0.

        Returns:
            Refined logits of shape (B, 1, H, W).
        """
        # Align spatial dimensions if needed
        if skip.shape[2:] != coarse_logits.shape[2:]:
            skip = F.interpolate(
                skip, size=coarse_logits.shape[2:],
                mode="bilinear", align_corners=False,
            )

        # Initialize state from [logits, skip]
        state_input = torch.cat([coarse_logits, skip], dim=1)
        state = self.state_init(state_input)

        # Run iterative NCA steps with gradient checkpointing
        remaining = self.n_steps
        while remaining > 0:
            chunk = min(self.checkpoint_every, remaining)
            if self.training and chunk > 1:
                # Use checkpoint to save VRAM during training
                state = checkpoint(
                    self._checkpointed_steps,
                    state,
                    chunk,
                    use_reentrant=False,
                )
            else:
                state = self._checkpointed_steps(state, chunk)
            remaining -= chunk

        # Readout: project hidden state → refined logits
        refined_logits = self.readout(state)
        return refined_logits
