"""CPU smoke test entrypoint for data -> model -> loss."""

from __future__ import annotations

import argparse

import torch

from .data import InfiniteRandomPatchDataset
from .fractal import LFDParams, compute_lfd_map
from .losses import dice_bce_loss
from .model import SimpleSwinUNet
from .models import FractalPriorSwinUNet, SwinUNetTiny
from .seed import set_deterministic_seed


def run_smoke_test(seed: int = 42, model_name: str = "simple", vessel_sampling: bool = False) -> float:
    """Run a deterministic smoke test on CPU.

    Returns:
        The scalar loss value.
    """

    set_deterministic_seed(seed)
    device = torch.device("cpu")

    image = torch.randn(3, 128, 128, device=device)
    mask = (torch.randn(1, 128, 128, device=device) > 0).float()

    sampling_config = None
    if vessel_sampling:
        sampling_config = {
            "enabled": True,
            "mode": "vessel_aware",
            "p_vessel": 0.7,
            "p_background": 0.3,
            "vessel_buffer": 3,
            "background_buffer": 0,
            "min_vessel_fraction_in_patch": 0.0,
            "max_retries": 30,
        }

    dataset = InfiniteRandomPatchDataset(
        image=image,
        mask=mask,
        patch_size=96,
        seed=seed,
        sampling_config=sampling_config,
        sample_id="smoke",
    )
    batch = next(iter(dataset))
    x = batch["image"].unsqueeze(0)
    y = batch["mask"].unsqueeze(0)

    lfd_map = compute_lfd_map(x[0], params=LFDParams())
    lfd_map = lfd_map.unsqueeze(0).unsqueeze(0)
    lfd_min = float(lfd_map.min().item())
    lfd_max = float(lfd_map.max().item())
    lfd_mean = float(lfd_map.mean().item())

    if model_name == "baseline":
        model = SwinUNetTiny(in_channels=x.shape[1]).to(device)
        logits = model(x)
    elif model_name == "fractal_prior":
        model = FractalPriorSwinUNet(in_channels=x.shape[1]).to(device)
        logits = model(x, lfd_map=lfd_map)
    else:
        model = SimpleSwinUNet(in_channels=x.shape[1]).to(device)
        logits = model(x, lfd_map=lfd_map)
    loss = dice_bce_loss(logits, y)

    print(f"LFD stats -> min: {lfd_min:.4f}, max: {lfd_max:.4f}, mean: {lfd_mean:.4f}")
    return float(loss.item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CPU smoke test.")
    parser.add_argument("--use_synth_data", action="store_true", help="Use synthetic data (default).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--model",
        type=str,
        default="simple",
        help="Model to run: simple|baseline|fractal_prior.",
    )
    parser.add_argument(
        "--vessel_sampling",
        type=str,
        default="off",
        choices=["on", "off"],
        help="Enable vessel-aware patch sampling.",
    )
    args = parser.parse_args()
    _ = args.use_synth_data

    loss_value = run_smoke_test(
        seed=args.seed,
        model_name=args.model,
        vessel_sampling=args.vessel_sampling == "on",
    )
    print(f"Smoke test loss: {loss_value:.6f}")


if __name__ == "__main__":
    main()
