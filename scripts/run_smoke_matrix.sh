#!/usr/bin/env bash
set -euo pipefail

python -m fractal_swin_unet.exp.run --matrix configs/matrices/smoke_matrix.yaml
