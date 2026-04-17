#!/usr/bin/env python3
"""Precompute PDE feature maps for all HRF images and save to disk.

This eliminates the ~10s/patch PDE computation from the training loop.
PDE maps are computed once per full image and cached as .pt files.
During training, patches are cropped from these cached maps.

Usage:
    python scripts/precompute_pde.py \
        --manifest manifests/hrf_full.jsonl \
        --out_dir cache/pde_maps \
        --dataset_root "/path/to/HRF"
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from fractal_swin_unet.pde import (
    perona_malik_diffusion,
    frangi_vesselness,
    meijering_neuriteness,
    fractional_laplacian,
)


def load_image(image_path: str) -> torch.Tensor:
    """Load image as (C, H, W) float tensor in [0, 1]."""
    from PIL import Image
    import numpy as np

    img = Image.open(image_path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)


def compute_all_pde_maps(
    image: torch.Tensor,
    device: torch.device | None = None,
    kappa: float = 30.0,
    n_iter: int = 15,
    frangi_sigmas: tuple[float, ...] = (1.0, 2.0, 4.0),
    meijering_sigmas: tuple[float, ...] = (0.5, 1.0, 2.0),
    frac_alpha: float = 0.5,
) -> torch.Tensor:
    """Compute all 4 PDE channels for a full image.

    Args:
        image: (3, H, W) tensor.
        device: Compute device (GPU if available for speedup).

    Returns:
        (4, H, W) tensor: [diffusion, frangi, meijering, frac_laplacian]
    """
    if device is not None and device.type == "cuda":
        image = image.to(device)

    channels = []

    # 1. Perona-Malik anisotropic diffusion
    diff = perona_malik_diffusion(image, kappa=kappa, n_iter=n_iter)
    if diff.ndim == 4:
        diff = diff.squeeze(0)
    if diff.shape[0] > 1:
        diff = diff[:1]
    channels.append(diff.squeeze(0).cpu())  # (H, W)

    # 2. Frangi vesselness
    frangi = frangi_vesselness(image, sigmas=frangi_sigmas)
    if frangi.ndim == 4:
        frangi = frangi.squeeze(0)
    channels.append(frangi.squeeze(0).cpu())

    # 3. Meijering neuriteness
    meij = meijering_neuriteness(image, sigmas=meijering_sigmas)
    if meij.ndim == 4:
        meij = meij.squeeze(0)
    channels.append(meij.squeeze(0).cpu())

    # 4. Fractional Laplacian
    frac = fractional_laplacian(image, alpha=frac_alpha)
    if frac.ndim == 4:
        frac = frac.squeeze(0)
    channels.append(frac.squeeze(0).cpu())

    return torch.stack(channels, dim=0)  # (4, H, W)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute PDE maps for HRF images.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="cache/pde_maps")
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--kappa", type=float, default=30.0)
    parser.add_argument("--n_iter", type=int, default=15)
    parser.add_argument("--frac_alpha", type=float, default=0.5)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load manifest
    with open(args.manifest, "r") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    print(f"Found {len(samples)} samples in manifest")

    total_time = 0.0
    for i, sample in enumerate(samples):
        sample_id = sample["id"]
        out_path = out_dir / f"{sample_id}.pt"

        if out_path.exists():
            print(f"  [{i+1}/{len(samples)}] {sample_id} — cached, skipping")
            continue

        # Resolve image path
        image_path = sample["image_path"]
        if args.dataset_root and not Path(image_path).is_absolute():
            image_path = str(Path(args.dataset_root) / image_path)

        print(f"  [{i+1}/{len(samples)}] {sample_id} — computing...", end="", flush=True)
        t0 = time.perf_counter()

        image = load_image(image_path)
        pde_maps = compute_all_pde_maps(image, device=device, kappa=args.kappa,
                                         n_iter=args.n_iter, frac_alpha=args.frac_alpha)

        # Save with metadata
        save_dtype = torch.float16 if args.dtype == "float16" else torch.float32
        torch.save({
            "pde_maps": pde_maps.to(save_dtype),  # (4, H, W)
            "channel_names": ["diffusion", "frangi", "meijering", "frac_laplacian"],
            "sample_id": sample_id,
            "shape": list(pde_maps.shape),
        }, out_path)

        elapsed = time.perf_counter() - t0
        total_time += elapsed
        print(f" {elapsed:.1f}s  shape={list(pde_maps.shape)}")

    print(f"\nDone! Total compute time: {total_time:.1f}s")
    print(f"Cache dir: {out_dir}")
    print(f"Files: {len(list(out_dir.glob('*.pt')))}")


if __name__ == "__main__":
    main()
