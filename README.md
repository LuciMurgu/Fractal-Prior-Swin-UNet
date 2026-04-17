# Fractal-Prior Swin-UNet

Lightweight, CPU-friendly scaffold for retinal vessel segmentation with a Fractal-Prior Swin-UNet variant.

## Install (uv)

```bash
uv venv
uv pip install -e .
```

## Smoke tests

```bash
python -m fractal_swin_unet.smoke_test
python -m fractal_swin_unet.smoke_test --model baseline
python -m fractal_swin_unet.smoke_test --model fractal_prior
```

## Train / eval / infer (synthetic data)

```bash
python -m fractal_swin_unet.train --config configs/smoke.yaml --use_synth_data
python -m fractal_swin_unet.eval  --config configs/smoke.yaml --use_synth_data
python -m fractal_swin_unet.infer --config configs/smoke.yaml --use_synth_data --out_dir runs/
```

## Fractal prior cache

```bash
python -m fractal_swin_unet.train --config configs/smoke_fractal_cache_disk.yaml --use_synth_data
python scripts/precompute_fractal_prior.py --use_synth_data --num_samples 10 --image_size 256 --cache_dir cache/fractal_prior
```

## Tiled full-image inference

```bash
python -m fractal_swin_unet.infer --config configs/smoke_tiled.yaml --use_synth_data --out_dir runs/
python -m fractal_swin_unet.eval  --config configs/smoke_tiled.yaml --use_synth_data
```

## Ablation smoke run

```bash
bash scripts/run_ablation_smoke.sh
```

## Datasets

This repo does not auto-download data. Provide a dataset root folder when the data layer is enabled (expected image/mask subfolders). See `docs/ARCHITECTURE.md` for the dataflow and integration points.
