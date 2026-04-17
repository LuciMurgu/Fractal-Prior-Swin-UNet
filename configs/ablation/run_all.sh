#!/bin/bash
# Run full ablation study — ~8-10 hours total on single GPU
# Each config: 40 epochs × 220 steps/epoch
# RESUME-AWARE: detects partial runs and resumes from the latest epoch checkpoint.
# Safe for nohup: use `nohup bash configs/ablation/run_all.sh &` to survive disconnects
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=0

# All 6 ablation configs (A-F)
CONFIGS=(
  A_no_prior
  B_fractal_only
  C_pde_only
  D_fractal_pde
  E_fractal_frangi_frac
  F_fractal_spade
)

echo "════════════════════════════════════════"
echo "Ablation study starting at $(date)"
echo "Project: $PROJECT_DIR"
echo "Configs: ${CONFIGS[*]}"
echo "════════════════════════════════════════"
echo ""

for config in "${CONFIGS[@]}"; do
  RUN_ID="ablation_${config}"
  RUN_DIR="runs/${RUN_ID}"
  LOG_FILE="runs/${RUN_ID}.log"

  # Skip if already completed (checkpoint.pt = final save after all epochs)
  if [ -f "${RUN_DIR}/checkpoint.pt" ]; then
    echo "⏭  Skipping $config — already completed (${RUN_DIR}/checkpoint.pt exists)"
    echo ""
    continue
  fi

  # Check for resumable epoch checkpoint (e.g. checkpoint_epoch_010.pt)
  RESUME_ARG=""
  LATEST_CKPT=""
  if [ -d "$RUN_DIR" ]; then
    LATEST_CKPT=$(ls -1v "${RUN_DIR}"/checkpoint_epoch_*.pt 2>/dev/null | tail -1)
    if [ -n "$LATEST_CKPT" ]; then
      RESUME_ARG="--resume_from ${LATEST_CKPT}"
      echo "🔄  Resuming $config from $(basename "$LATEST_CKPT")"
    fi
  fi

  echo "════════════════════════════════════════"
  echo "▶  Running: $config ($(date))"
  echo "   Log: $LOG_FILE"
  if [ -n "$RESUME_ARG" ]; then
    echo "   Resume: $LATEST_CKPT"
  fi
  echo "════════════════════════════════════════"

  # Clear CUDA cache between configs to prevent memory fragmentation
  python3 -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true

  python -m fractal_swin_unet.train \
    --config "configs/ablation/${config}.yaml" \
    --run_id "${RUN_ID}" \
    --require_gpu \
    $RESUME_ARG \
    2>&1 | tee -a "$LOG_FILE"

  EXIT_CODE=${PIPESTATUS[0]}
  if [ $EXIT_CODE -ne 0 ]; then
    echo "✗  FAILED: $config (exit code $EXIT_CODE) at $(date)"
    echo "   Check $LOG_FILE for details"
    echo "   Continuing with next config..."
    echo ""
    continue
  fi

  echo "✓  Completed: $config ($(date))"
  echo ""
done

echo ""
echo "════════════════════════════════════════"
echo "All ablation training complete at $(date)"
echo ""
echo "Next steps:"
echo "  python scripts/run_ablation_eval.py --ablation-dir runs/"
echo "════════════════════════════════════════"
