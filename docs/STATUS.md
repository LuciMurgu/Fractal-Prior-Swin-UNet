# Fractal-Prior Swin-UNet — Status (Engineering Simulation → Hardening + Dataset Onboarding)

This repo is in **Hardening + Real Dataset Onboarding + Evidence Pack** phase.  
Core synth smoke must remain green at all times.

## Quick Commands
- Tests: `pytest -q`
- Smoke: `python -m fractal_swin_unet.smoke_test --use_synth_data`
- Train: `python -m fractal_swin_unet.train --config configs/smoke.yaml --use_synth_data`
- Eval:  `python -m fractal_swin_unet.eval  --config configs/smoke.yaml --use_synth_data`
- Infer: `python -m fractal_swin_unet.infer --config configs/smoke.yaml --use_synth_data --out_dir runs/`

## Global DoD — Tier 1 Core (always green)
- [x] `pytest -q`
- [x] `python -m fractal_swin_unet.smoke_test --use_synth_data`
- [x] `python -m fractal_swin_unet.train --config configs/smoke.yaml --use_synth_data`
- [x] `python -m fractal_swin_unet.eval  --config configs/smoke.yaml --use_synth_data`
- [x] `python -m fractal_swin_unet.infer --config configs/smoke.yaml --use_synth_data --out_dir runs/`

## Global DoD — Tier 2 Full-image correctness (if enabled)
- [x] `python -m fractal_swin_unet.infer --config configs/smoke_tiled.yaml --use_synth_data --out_dir runs/`
- [x] `python -m fractal_swin_unet.eval  --config configs/smoke_tiled.yaml --use_synth_data`

## Global DoD — Tier 3 Reproducibility contract
- [ ] run dir contains: resolved_config + code_hash + manifest_hash + metrics.json
- [ ] threshold sweep: tau* selected from VAL only; report BestDice and Dice@0.5

## Global DoD — Tier 4 HRF onboarding
- [x] `python scripts/make_hrf_manifest.py --root "/home/lucian/Retina Datasets/HRF" --out manifests/hrf.jsonl`
- Full dataset (masks + roi_masks layout): `python scripts/make_hrf_manifest.py --root "/home/lucian/medical-imaging-ai/Retina Datasets/Full retina dataset/HRF" --manual_subdir masks --fov_subdir roi_masks --out manifests/hrf_full.jsonl` then `python -m fractal_swin_unet.train --config configs/hrf_full.yaml`
- [x] `python -m fractal_swin_unet.train --config configs/hrf_smoke_cpu.yaml`
- [x] `python -m fractal_swin_unet.eval  --config configs/hrf_smoke_cpu.yaml` (FOV-masked default)
- [x] `python -m fractal_swin_unet.infer --config configs/hrf_smoke_cpu.yaml --out_dir runs/`

## Global DoD — Tier 5 Evidence pack (matrix runner)
- [ ] `python -m fractal_swin_unet.exp.run --matrix configs/matrices/smoke_matrix.yaml`
- [ ] HRF baseline vs fractal ablation: `python -m fractal_swin_unet.exp.run --matrix configs/matrices/hrf_ablation_matrix.yaml` (requires HRF manifest + data paths)
- [ ] reports generated: CSV + MD + JSON + qualitative PNGs

---

# Phase P0 — Repo scaffold & tooling
- [ ] uv/pyproject locked deps
- [ ] package layout under src/
- [ ] entrypoints exist (smoke_test/train/eval/infer)
- [ ] AGENTS.md present
- [ ] `pytest -q` on fresh clone

# Phase P1 — Data layer (synth + real-path mode)
- [ ] InfinitePatchDataset random crop each call
- [ ] deterministic seed option
- [ ] lightweight augs (flip/rotate) applied consistently
- [ ] synth dataset supports full-image and patch modes
- [ ] tests for patching/transforms

# Phase P2 — Fractal prior core (DBC→LFD)
- [ ] DBC implemented
- [ ] LFD map fast mode (coarse grid + upsample)
- [ ] normalized prior [0,1]
- [ ] tests for LFD range/shape/stability

# Phase P3 — Models baseline
- [ ] baseline Swin-UNet forward logits (B,1,H,W)
- [ ] forward shape tests
- [ ] smoke baseline runs

# Phase P4 — Fractal gating integration
- [ ] skip gating: skip *= sigmoid(alpha * norm(lfd_stage))
- [ ] multi-scale downsampling of LFD
- [ ] ablation toggle for fractal gate
- [ ] gating tests

# Phase P5 — Train/Eval/Infer pipelines
- [ ] loss default Dice+BCE
- [ ] optional focal/clDice toggles
- [ ] checkpointing + seeds
- [ ] infer writes PNG outputs
- [ ] run artifacts written (if enabled)

# Phase P6 — Full-image tiled inference + FOV-masked eval
- [ ] tiled infer engine (overlap + blending)
- [ ] full-image eval (not per-tile)
- [ ] FOV mask handling (bool, resize nearest if needed)
- [ ] tests for tiling + mask metrics

# Phase P7 — HRF dataset onboarding
- [ ] HRF manifest generator + stem matching
- [ ] HRF dataset adapter (image, vessel mask, fov mask)
- [ ] HRF configs (train/eval/infer) using tiled infer and FOV-masked eval
- [ ] smoke scripts for HRF
- [ ] tests for manifest matching + binarization

# Phase P8 — Evidence pack + matrix runner
- [ ] matrix YAML format + deep overrides
- [ ] runner generates multiple runs deterministically
- [ ] reports aggregator CSV/MD/JSON
- [ ] qualitative panels export
- [ ] docs/EXPERIMENTS.md

---

## Milestone Log (keep short)
| Date (UTC) | Milestone | Result | Notes |
|---|---|---|---|
| 2026-02-10 | Tier 1 core (partial) | PASS | pytest + smoke_test green |
| 2026-02-10 | Tier 1 core | PASS | pytest + smoke_test + train/eval/infer green |
| 2026-02-10 | Tier 4 HRF smoke | PASS | manifest + train/eval/infer green |
| 2026-02-13 | Tier 1 re-validation after wavelet-gated integration | PASS | pytest + smoke_test + synth train/eval/infer (`wavelet_tier1_*`) green |
| 2026-02-14 | Tier 1 re-validation after Fractal v2 (config-gated) | PASS | pytest + smoke_test + synth train/eval/infer (`fractal_v2_tier1_*`) green |
| 2026-04-06 | V7 NCA module integration (config-gated) | PASS | `fractal_nca.py` + 7 unit tests + smoke_nca.yaml train/eval/infer green. Tier 1 core unaffected (50 tests pass). |
| 2026-04-07 | Tier 2 Full-image correctness for HRF | PASS | Created `hrf_tiled.yaml`. Sliding-window overlapping inference generated correctly without OOM. |
| 2026-04-07 | Full Tier 1+2 re-validation (fresh restart) | PASS | 46/46 tests pass. smoke_test/train/eval/infer green. Tiled infer+eval green (BestDice 0.66). |