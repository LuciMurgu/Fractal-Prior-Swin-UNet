#!/usr/bin/env bash
# run_pending_experiments.sh — Sequential experiment chain
# Runs: BASELINE_200ep (resume) → OPT_F_lacunarity → OPT_G_percolation
# Each job completes train+eval+infer before the next starts.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON=".venv/bin/python"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_experiment() {
    local CONFIG="$1"
    local RUN_ID="$2"
    local EXTRA_ARGS="${3:-}"

    log "═══ Starting $RUN_ID (config: $CONFIG) ═══"

    # Train (with resume support)
    log "Training $RUN_ID..."
    $PYTHON -m fractal_swin_unet.train \
        --config "$CONFIG" \
        --run_id "$RUN_ID" \
        $EXTRA_ARGS 2>&1 | tee "runs/${RUN_ID}.log"

    # Eval
    log "Evaluating $RUN_ID..."
    $PYTHON -m fractal_swin_unet.eval \
        --config "$CONFIG" \
        --checkpoint "runs/${RUN_ID}/checkpoint_best.pt" \
        --run_id "${RUN_ID}_eval" 2>&1 | tee -a "runs/${RUN_ID}.log"

    # Infer
    log "Inference $RUN_ID..."
    $PYTHON -m fractal_swin_unet.infer \
        --config "$CONFIG" \
        --checkpoint "runs/${RUN_ID}/checkpoint_best.pt" \
        --out_dir "runs/${RUN_ID}_infer" 2>&1 | tee -a "runs/${RUN_ID}.log"

    log "═══ Completed $RUN_ID ═══"
    echo ""
}

# ─── Job 1: BASELINE 200ep (resume from epoch 42) ───
log "Checking baseline_200ep status..."
BASELINE_EPOCH=$($PYTHON -c "
import torch
try:
    ckpt = torch.load('runs/baseline_200ep/checkpoint_latest.pt', map_location='cpu', weights_only=False)
    print(ckpt.get('epoch', 0))
except: print(0)
" 2>/dev/null)
log "Baseline at epoch ${BASELINE_EPOCH}/200"

if [ "${BASELINE_EPOCH}" -lt 200 ]; then
    run_experiment \
        "configs/optim/BASELINE_200ep.yaml" \
        "baseline_200ep" \
        "--resume_from runs/baseline_200ep/checkpoint_latest.pt"
else
    log "Baseline already at 200 epochs, skipping training."
    # Still run eval+infer
    $PYTHON -m fractal_swin_unet.eval \
        --config "configs/optim/BASELINE_200ep.yaml" \
        --checkpoint "runs/baseline_200ep/checkpoint_best.pt" \
        --run_id "baseline_200ep_eval" 2>&1 | tee "runs/baseline_200ep_eval.log"
fi

# ─── Job 2: OPT-F lacunarity (100ep from scratch) ───
run_experiment \
    "configs/optim/OPT_F_lacunarity.yaml" \
    "opt_F_lacunarity"

# ─── Job 3: OPT-G percolation (100ep from scratch) ───
run_experiment \
    "configs/optim/OPT_G_percolation.yaml" \
    "opt_G_percolation"

log "═══ ALL EXPERIMENTS COMPLETE ═══"
log "Next: run scripts/compile_journal_table.py to generate reports."
