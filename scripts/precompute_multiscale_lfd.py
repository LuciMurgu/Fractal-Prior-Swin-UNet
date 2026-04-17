#!/usr/bin/env python3
"""Precompute multi-scale LFD + lacunarity maps for all FIVES images.

This is a ONE-TIME operation that eliminates the per-patch CPU bottleneck
during training. Saves a 4-channel (H, W) tensor per image:
  - Ch 0: LFD at window_size=16  (fine: capillaries)
  - Ch 1: LFD at window_size=32  (medium: small vessels)
  - Ch 2: LFD at window_size=64  (coarse: main arteries)
  - Ch 3: Lacunarity map          (texture heterogeneity)

Usage:
    python scripts/precompute_multiscale_lfd.py \
        --manifest manifests/fives.jsonl \
        --dataset_root "/home/lucian/medical-imaging-ai/Retina Datasets/Full retina dataset/FIVES" \
        --out_dir cache/multiscale_lfd \
        --workers 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

# Add project to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fractal_swin_unet.fractal.lfd import LFDParams, compute_lfd_map
from fractal_swin_unet.fractal.lacunarity import compute_lacunarity_map


WINDOW_SIZES = (16, 32, 64)
BOX_SIZES = (2, 4, 8)
LACUNARITY_WINDOW = 16  # Small window for speed
LACUNARITY_BOX_SIZES = (4, 8)


def _load_image(image_path: str) -> torch.Tensor:
    """Load image as (C, H, W) float32 tensor in [0, 1]."""
    from PIL import Image
    import torchvision.transforms.functional as TF
    img = Image.open(image_path).convert("RGB")
    return TF.to_tensor(img)


def _compute_one(
    sample_id: str,
    image_path: str,
    out_dir: Path,
) -> dict:
    """Compute multi-scale LFD + lacunarity for one image."""
    out_path = out_dir / f"{sample_id}.pt"
    if out_path.exists():
        return {"id": sample_id, "status": "cached", "path": str(out_path)}

    t0 = time.time()
    image = _load_image(image_path)
    gray = image.mean(dim=0)  # (H, W)
    
    channels = []
    
    # Multi-scale LFD
    for ws in WINDOW_SIZES:
        # Use box_sizes that fit within the window
        valid_boxes = tuple(b for b in BOX_SIZES if b < ws)
        if not valid_boxes:
            valid_boxes = (2,)
        params = LFDParams(
            window_size=ws,
            box_sizes=valid_boxes,
            fast_mode=True,
        )
        lfd = compute_lfd_map(gray, params=params)
        channels.append(lfd)
    
    # Lacunarity  
    lac = compute_lacunarity_map(
        gray,
        window_size=LACUNARITY_WINDOW,
        box_sizes=LACUNARITY_BOX_SIZES,
    )
    channels.append(lac)
    
    # Stack: (4, H, W)
    stack = torch.stack(channels, dim=0).half()  # fp16 to save disk
    
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"maps": stack, "id": sample_id, "channels": [
        f"lfd_w{ws}" for ws in WINDOW_SIZES
    ] + ["lacunarity"]}, out_path)
    
    elapsed = time.time() - t0
    h, w = gray.shape
    return {
        "id": sample_id,
        "status": "computed",
        "path": str(out_path),
        "shape": f"{h}x{w}",
        "time_s": round(elapsed, 1),
        "size_mb": round(out_path.stat().st_size / 1024 / 1024, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Precompute multi-scale LFD maps")
    parser.add_argument("--manifest", required=True, help="Path to JSONL manifest")
    parser.add_argument("--dataset_root", required=True, help="Dataset root directory")
    parser.add_argument("--out_dir", default="cache/multiscale_lfd", help="Output directory")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (careful with RAM)")
    parser.add_argument("--max_images", type=int, default=0, help="Limit images (0=all)")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)

    # Load manifest
    samples = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if args.max_images > 0:
        samples = samples[:args.max_images]

    print(f"Manifest: {manifest_path} ({len(samples)} images)")
    print(f"Output: {out_dir}")
    print(f"Window sizes: {WINDOW_SIZES}")
    print(f"Workers: {args.workers}")
    print()

    # Check how many are already cached
    already_cached = sum(
        1 for s in samples
        if (out_dir / f"{s['id']}.pt").exists()
    )
    if already_cached:
        print(f"Already cached: {already_cached}/{len(samples)}")

    t_start = time.time()
    results = []

    if args.workers <= 1:
        # Sequential — simpler, less RAM
        for i, sample in enumerate(samples):
            image_path = str(dataset_root / sample["image_path"])
            result = _compute_one(sample["id"], image_path, out_dir)
            results.append(result)
            status = result["status"]
            if status == "computed":
                print(
                    f"[{i+1}/{len(samples)}] {sample['id']}: "
                    f"{result['shape']} in {result['time_s']}s "
                    f"({result['size_mb']}MB)"
                )
            else:
                print(f"[{i+1}/{len(samples)}] {sample['id']}: {status}")
    else:
        # Parallel
        futures = {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for sample in samples:
                image_path = str(dataset_root / sample["image_path"])
                future = pool.submit(_compute_one, sample["id"], image_path, out_dir)
                futures[future] = sample["id"]

            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                results.append(result)
                sid = futures[future]
                if result["status"] == "computed":
                    print(
                        f"[{i+1}/{len(samples)}] {sid}: "
                        f"{result.get('shape', '?')} in {result.get('time_s', '?')}s"
                    )
                else:
                    print(f"[{i+1}/{len(samples)}] {sid}: {result['status']}")

    elapsed = time.time() - t_start
    computed = sum(1 for r in results if r["status"] == "computed")
    cached = sum(1 for r in results if r["status"] == "cached")
    
    total_size = sum(
        Path(r["path"]).stat().st_size 
        for r in results 
        if Path(r.get("path", "")).exists()
    ) / 1024 / 1024

    print(f"\n{'='*60}")
    print(f"DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Computed: {computed}, Cached: {cached}, Total: {len(results)}")
    print(f"Disk usage: {total_size:.1f} MB")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
