#!/usr/bin/env bash
# Launch OPT-F v2 (lacunarity) and OPT-G v2 (percolation) training
# clDice-optimized with fixed lacunarity/percolation signal
# Expected runtime: ~7-8 hours each (150 epochs × 400 steps)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

PYTHON=".venv/bin/python3"
LOG="logs/train_optfg_v2.log"
mkdir -p logs

echo "═══════════════════════════════════════════" | tee -a "$LOG"
echo "  OPT-F v2 + OPT-G v2 Training Launch"      | tee -a "$LOG"
echo "  Started: $(date)"                           | tee -a "$LOG"
echo "═══════════════════════════════════════════" | tee -a "$LOG"

# ── OPT-F v2: LFD + Lacunarity ──────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[1/2] OPT-F v2: LFD + Lacunarity (150 epochs, clDice-optimized)" | tee -a "$LOG"
echo "  Config: configs/optim/OPT_F_lacunarity_v2.yaml" | tee -a "$LOG"
echo "  Started: $(date)" | tee -a "$LOG"

$PYTHON -m fractal_swin_unet.train \
    --config configs/optim/OPT_F_lacunarity_v2.yaml \
    --run_id opt_F_lacunarity_v2 \
    2>&1 | tee -a "$LOG"

echo "  OPT-F v2 finished: $(date)" | tee -a "$LOG"

# ── OPT-G v2: LFD + Lacunarity + Percolation ────────────────────────
echo "" | tee -a "$LOG"
echo "[2/2] OPT-G v2: LFD + Lacunarity + Percolation (150 epochs)" | tee -a "$LOG"
echo "  Config: configs/optim/OPT_G_percolation_v2.yaml" | tee -a "$LOG"
echo "  Started: $(date)" | tee -a "$LOG"

$PYTHON -m fractal_swin_unet.train \
    --config configs/optim/OPT_G_percolation_v2.yaml \
    --run_id opt_G_percolation_v2 \
    2>&1 | tee -a "$LOG"

echo "  OPT-G v2 finished: $(date)" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "═══════════════════════════════════════════" | tee -a "$LOG"
echo "  ALL TRAINING COMPLETE: $(date)"             | tee -a "$LOG"
echo "═══════════════════════════════════════════" | tee -a "$LOG"
