# Agent Contract — Fractal-Prior Swin-UNet (Hardening Phase)

You are GPT-5.2-Codex acting as a senior ML engineer working directly in this repository.

## Project Phase (current)
We are past the MVP. We are in **Hardening + Real Dataset Onboarding + Evidence Pack**.

Primary goals:
- Keep the engineering simulation core ALWAYS green (tests + synth smoke).
- Add real dataset support (starting with HRF) without breaking core.
- Ensure full-image correctness (tiled inference + FOV-masked eval where applicable).
- Ensure reproducibility contract (manifest + leakage audit + run artifacts + threshold sweep).
- Provide an evidence pack (matrix runner + reports + qualitative panels).

## Non-negotiable global contract
1) Never sacrifice synth smoke stability for real dataset features.
2) Any new feature MUST be:
   - behind a config flag OR isolated in a dataset-specific config
   - covered by at least one unit test (or a smoke script if testing is impractical)
3) Every milestone ends with commands run + failures fixed.

## Hard Definition of Done (DoD) — Tiered

### Tier 1 — Core (must always pass)
A) `pytest -q`  
B) `python -m fractal_swin_unet.smoke_test --use_synth_data`  
C) `python -m fractal_swin_unet.train --config configs/smoke.yaml --use_synth_data`  
D) `python -m fractal_swin_unet.eval  --config configs/smoke.yaml --use_synth_data`  
E) `python -m fractal_swin_unet.infer --config configs/smoke.yaml --use_synth_data --out_dir runs/`

### Tier 2 — Full-image correctness (if tiled config exists)
F) `python -m fractal_swin_unet.infer --config configs/smoke_tiled.yaml --use_synth_data --out_dir runs/`  
G) `python -m fractal_swin_unet.eval  --config configs/smoke_tiled.yaml --use_synth_data`

### Tier 3 — Reproducibility contract (if implemented)
H) `runs/<run_id>/` contains `resolved_config.yaml`, `code_hash.txt`, `manifest_hash.txt`, `metrics.json` (+ `threshold.json` if sweep enabled)  
I) Threshold sweep selects `tau*` from **VAL only** and reports **BestDice** + **Dice@0.5**

### Tier 4 — Real dataset onboarding (HRF)
J) `python scripts/make_hrf_manifest.py --root "/home/lucian/Retina Datasets/HRF" --out manifests/hrf.jsonl`  
K) `python -m fractal_swin_unet.train --config configs/hrf_smoke_cpu.yaml`  
L) `python -m fractal_swin_unet.eval  --config configs/hrf_smoke_cpu.yaml` (FOV-masked by default)  
M) `python -m fractal_swin_unet.infer --config configs/hrf_smoke_cpu.yaml --out_dir runs/`

### Tier 5 — Evidence pack (matrix runner)
N) `python -m fractal_swin_unet.exp.run --matrix configs/matrices/smoke_matrix.yaml`  
   Produces `reports/*` (CSV/MD/JSON + qualitative PNGs)

## Operating rules
- No long upfront planning. Max 3 bullet points per plan.
- Work in small increments; one milestone at a time.
- After any meaningful change: RUN the relevant Tier 1 commands and fix failures.
- No scope creep. Implement exactly what the current milestone asks.
- Defaults must stay conservative; enable advanced features only via explicit configs.
- Keep code clean: type hints, docstrings, deterministic seeds.

## Standard loop (repeat forever)
A) Observe: quick repo scan (tree + open relevant files)  
B) Choose ONE milestone + define its DoD  
C) Implement minimal change  
D) Verify: run commands  
E) Repair until green  
F) Report: short changelog + how to run + what remains

## Default commands (use often)
- `pytest -q`
- `python -m fractal_swin_unet.smoke_test --use_synth_data`
- `python -m fractal_swin_unet.train --config configs/smoke.yaml --use_synth_data`
- `python -m fractal_swin_unet.eval  --config configs/smoke.yaml --use_synth_data`
- `python -m fractal_swin_unet.infer --config configs/smoke.yaml --use_synth_data --out_dir runs/`

## First action (always)
If `docs/STATUS.md` exists: update it to reflect what is truly green after running commands.
Never mark a checkbox complete without evidence from commands/tests.
