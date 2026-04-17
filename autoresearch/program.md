# AutoResearch: Fractal-Prior Vessel Segmentation

## Goal
Maximize `best_full_eval_dice` on HRF retinal vessel segmentation.

## Current Best
- **Config H**: Dice = 0.8096 (epoch 60, τ*=0.55)
- Architecture: Standard Swin-UNet + Fractal Anisotropic Diffusion PDE + 5-ch SPADE v2
- Loss: Dice + BCE + Focal + clDice + Skeleton Recall + Fractal-weighted BCE

## Constraints
- Single GPU: RTX 3060 (8GB VRAM)
- Each experiment: 20 epochs max (~30 min)
- Dataset: HRF (45 images, 3504×2336)
- Must not crash or OOM
- Must use existing train.py infrastructure

## Search Space (what can be changed)
### Architecture
- `embed_dim`: [32, 48, 64] — model capacity
- `depths`: [[1,1,1], [2,1,1], [2,2,1]] — encoder depth
- `spade_hidden`: [16, 32, 48] — SPADE v2 hidden dim
- `fad_n_steps`: [3, 5, 7, 10] — PDE Euler steps

### Training
- `lr`: [2e-4, 3e-4, 5e-4, 8e-4, 1e-3] — learning rate
- `weight_decay`: [0.001, 0.005, 0.01, 0.02] — regularization
- `steps_per_epoch`: [220, 300, 400] — training volume
- `grad_clip_norm`: [0.5, 1.0, 2.0] — gradient clipping

### Loss Weights
- `skeleton_recall.weight`: [0.05, 0.10, 0.15, 0.20, 0.30]
- `fractal_bce.weight`: [0.05, 0.10, 0.15, 0.20]
- `fractal_bce.alpha`: [0.5, 1.0, 1.5, 2.0, 3.0]
- `cldice.weight`: [0.05, 0.10, 0.12, 0.15, 0.20]
- `focal.weight`: [0.05, 0.10, 0.15, 0.20]

### Data
- `patch_size`: [256, 384, 512]
- `p_vessel`: [0.6, 0.7, 0.8]
- `photometric_aug.brightness`: [0.05, 0.08, 0.12]
- `photometric_aug.contrast`: [0.10, 0.15, 0.20]

## Rules
1. Each experiment mutates 1-2 params from current best config
2. 20-epoch training with full eval at epoch 10 and 20
3. If best_full_eval_dice > current_best → adopt as new baseline
4. If worse → revert, try different mutation
5. Log everything to leaderboard.json
