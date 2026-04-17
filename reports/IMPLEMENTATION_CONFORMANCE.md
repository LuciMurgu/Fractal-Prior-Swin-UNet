# Implementation Conformance Report — Fractal-Prior Swin-UNet

## 1) Snapshot
- Date/time: 2026-02-10
- Git commit: unknown (no .git found)
- Python: 3.13.7
- Dependencies: torch=2.10.0+cu128, PyYAML=6.0.3, numpy=2.4.2

## 2) Global DoD Status: PASS
All required commands executed successfully in this audit run.

## 3) Spec Checklist (PASS/FAIL + Evidence)
| Item | Status | Evidence |
|---|---|---|
| Baseline Swin-UNet binary segmentation | PASS | `src/fractal_swin_unet/models/swin_unet.py` exposes `SwinUNetTiny` |
| Fractal prior from image only (DBC→LFD, grayscale) | PASS | `src/fractal_swin_unet/fractal/engine.py`, `fractal/provider.py` uses image-only grayscale |
| Fractal gating on skip connections | PASS | `src/fractal_swin_unet/models/fractal_prior_swin_unet.py` uses `apply_fractal_gate` on skips |
| Infinite patch training + optional vessel-aware sampling | PASS | `src/fractal_swin_unet/data/__init__.py`, `data/sampling.py` |
| Loss defaults Dice+BCE, optional focal+clDice | PASS | `src/fractal_swin_unet/losses.py`, `configs/smoke.yaml` (focal/cldice disabled) |
| Full-image tiled inference + seam-free blending | PASS | `src/fractal_swin_unet/inference/tiling.py`, `configs/smoke_tiled.yaml` |
| FOV-aware eval (optional) | PASS | `src/fractal_swin_unet/metrics.py` supports mask; `data/fov.py` loader |
| Reproducibility contract (manifest, splits, artifacts, sweep) | PASS | `src/fractal_swin_unet/exp/*`, run dirs under `runs/` |
| Threshold sweep from val only | PASS | `src/fractal_swin_unet/exp/threshold.py`, `src/fractal_swin_unet/eval.py` |
| Evidence pack (matrix runner + reports + panels) | PASS | `src/fractal_swin_unet/exp/run.py`, `reports/smoke_matrix_*` |

## 4) Drift / Discrepancies
- **docs/STATUS.md mismatch**: STATUS lists only 3 milestones, but repo contains additional implemented features (loss toggles, tiled infer/eval, fractal provider, matrix runner). This doc is out of date relative to code and reports.
- **FOV mask loading**: `data/fov.py` currently loads masks via `torch.load` only; spec suggests PNG support. This is a partial implementation.

## 5) Risks
- **Eval loss uses `torch.logit` on probabilities**: `eval.py` computes `val_loss = dice_bce_loss(val_probs.logit(), y)` which can be unstable if probabilities reach 0 or 1. This is not a spec requirement, but could cause NaNs in edge cases.
- **STATUS drift**: If STATUS is used for external reporting, it understates implemented features.

## 6) Actionable Next Steps (Minimal)
1. Update `docs/STATUS.md` to reflect implemented milestones (loss toggles, tiled infer/eval, fractal prior provider, matrix runner).
2. Extend `data/fov.py` to support PNG FOV masks (if needed for real datasets).
3. Replace `val_probs.logit()` with logits when available in eval to avoid instability (if observed).

## Command Evidence
- `pytest -q`:
  - PASS (29 tests)
- `python -m fractal_swin_unet.smoke_test --use_synth_data`:
  - Output: `Smoke test loss: 1.179274`
- `python -m fractal_swin_unet.train --config configs/smoke.yaml --use_synth_data`:
  - Output: `FINAL_METRICS train_dice=0.6091 val_dice=0.5839`
- `python -m fractal_swin_unet.eval --config configs/smoke.yaml --use_synth_data`:
  - Output: `BestDice 0.6668 Dice@0.5 0.6666 tau* 0.10`
- `python -m fractal_swin_unet.infer --config configs/smoke_tiled.yaml --use_synth_data --out_dir runs/`:
  - Output: `Saved predictions to runs/20260210_114934_892765/preds`
- `python -m fractal_swin_unet.exp.run --matrix configs/matrices/smoke_matrix.yaml`:
  - Reports generated under `reports/` and qualitative PNGs under `reports/smoke_matrix_qual/`
