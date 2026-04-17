#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lucian/Retina Datasets/HRF"
MANIFEST="manifests/hrf.jsonl"

PYTHON_BIN="python"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

if [[ ! -f "$MANIFEST" ]]; then
  "$PYTHON_BIN" scripts/make_hrf_manifest.py --root "$ROOT" --out "$MANIFEST"
fi

"$PYTHON_BIN" -m fractal_swin_unet.train --config configs/hrf_smoke_cpu.yaml
"$PYTHON_BIN" -m fractal_swin_unet.eval  --config configs/hrf_smoke_cpu.yaml
"$PYTHON_BIN" -m fractal_swin_unet.infer --config configs/hrf_smoke_cpu.yaml --out_dir runs/
