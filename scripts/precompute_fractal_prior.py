"""Precompute fractal prior maps for a dataset or synthetic data."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from fractal_swin_unet.fractal.provider import FractalPriorConfig, FractalPriorProvider


def _make_synth_image(sample_id: str, image_size: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    image = torch.randn(3, image_size, image_size, generator=gen)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute fractal prior maps.")
    parser.add_argument("--use_synth_data", action="store_true", help="Use synthetic data.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of synthetic samples.")
    parser.add_argument("--image_size", type=int, default=256, help="Synthetic image size.")
    parser.add_argument("--cache_dir", type=str, default="cache/fractal_prior", help="Cache directory.")
    args = parser.parse_args()

    config = FractalPriorConfig(
        enabled=True,
        caching_enabled=True,
        caching_mode="disk",
        cache_dir=args.cache_dir,
        cache_write=True,
        cache_read=True,
        precompute_enabled=True,
    )
    provider = FractalPriorProvider(config)

    total = args.num_samples
    hits = 0
    start = time.perf_counter()

    for i in range(total):
        sample_id = f"synth_{i}"
        image = _make_synth_image(sample_id, args.image_size, seed=1234 + i)
        cache_key = provider._cache_key(sample_id)
        cache_file = Path(args.cache_dir) / f"{cache_key}.pt"
        if cache_file.exists():
            hits += 1
        _ = provider.get_full(sample_id, image)

    elapsed = time.perf_counter() - start
    hit_rate = hits / total if total else 0.0
    print(f"Precompute completed in {elapsed:.2f}s, cache hit rate {hit_rate:.2%}")


if __name__ == "__main__":
    main()
