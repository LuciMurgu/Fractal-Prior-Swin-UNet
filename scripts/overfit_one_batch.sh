#!/usr/bin/env bash
set -euo pipefail

python -m fractal_swin_unet.train --config configs/smoke.yaml --use_synth_data --overfit_one_batch "$@"
