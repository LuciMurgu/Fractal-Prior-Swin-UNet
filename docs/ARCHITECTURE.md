# Architecture Overview

## Components
- `fractal_swin_unet/fractal`: DBC + LFD map generation utilities.
- `fractal_swin_unet/models`: Swin-UNet baseline and Fractal-Prior variant with gating.
- `fractal_swin_unet/data`: Infinite patch sampling + synthetic data utilities.
- `fractal_swin_unet/train|eval|infer`: CPU-friendly pipelines driven by YAML configs.

## Dataflow (train)
1. Load YAML config and apply CLI overrides.
2. Seed all RNGs for reproducibility.
3. Build model (baseline or fractal prior).
4. Sample synthetic patch batch (or dataset in future).
5. (Fractal mode) Compute LFD map and downsample to skip resolutions.
6. Forward pass → Dice+BCE loss → optimizer step.
7. Save checkpoint; report final train/val Dice.

## Fractal gating
For each encoder skip, a downsampled LFD map is normalized and passed through a sigmoid gate:

```
normalized = (lfd - mean) / (std + eps)
gate = sigmoid(alpha * normalized)
skip = skip * gate
```
