#!/usr/bin/env bash
# ============================================================
# Overnight Parameter Optimization Sweep
# RTX 4070 Laptop (8GB VRAM) — sequential, crash-safe
#
# Runs OPT-A through OPT-D on FIVES, clDice-prioritized.
# Each run: train → eval → cleanup → next
#
# Usage:
#   nohup bash scripts/run_optim_sweep.sh > logs/optim_sweep.log 2>&1 &
# ============================================================

set -euo pipefail

PYTHON="python3"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
source .venv/bin/activate 2>/dev/null || true

RUN_BASE="runs"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

CONFIGS=(
    "configs/optim/OPT_A_more_pde_steps.yaml|opt_A_pde15"
    "configs/optim/OPT_B_finer_lfd.yaml|opt_B_finer_lfd"
    "configs/optim/OPT_C_bigger_spade_skel.yaml|opt_C_spade64_skel"
    "configs/optim/OPT_D_longer_training.yaml|opt_D_longer"
)

echo "============================================================"
echo "  OPTIM SWEEP — $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Configs: ${#CONFIGS[@]}"
echo "============================================================"

RESULTS_FILE="$LOG_DIR/optim_sweep_results.csv"
echo "config,run_id,best_dice,best_cldice,tau_star,train_dice,val_dice,status" > "$RESULTS_FILE"

for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r config_path run_id <<< "$entry"
    run_dir="${RUN_BASE}/${run_id}"

    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "  CONFIG: $config_path"
    echo "  RUN ID: $run_id"
    echo "  START:  $(date)"
    echo "────────────────────────────────────────────────────────────"

    # --- Skip if already completed ---
    if [ -f "$run_dir/metrics.json" ]; then
        echo "  ⏭  SKIP: $run_id already has metrics.json"
        # Extract results from existing run
        $PYTHON -c "
import json
m = json.load(open('$run_dir/metrics.json'))
dice = m.get('best_full_eval_dice', m.get('val_dice', 0))
tau = m.get('best_full_eval_tau_star', '?')
td = m.get('train_dice', '?')
vd = m.get('val_dice', '?')
print(f'$run_id,{run_id},{dice},{tau},{td},{vd},skipped')
" >> "$RESULTS_FILE" 2>/dev/null || echo "$run_id,$run_id,?,?,?,?,skip_error" >> "$RESULTS_FILE"
        continue
    fi

    # --- Train ---
    echo "  ▶ Training..."
    if $PYTHON -m fractal_swin_unet.train \
        --config "$config_path" \
        --run_id "$run_id" \
        --run_base "$RUN_BASE" \
        --require_gpu \
        2>&1 | tee "$LOG_DIR/${run_id}_train.log" | tail -5; then
        echo "  ✅ Train completed"
    else
        echo "  ❌ Train FAILED — skipping eval"
        echo "$run_id,$run_id,FAIL,FAIL,FAIL,FAIL,FAIL,train_failed" >> "$RESULTS_FILE"
        # Cleanup GPU memory
        $PYTHON -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        sleep 5
        continue
    fi

    # --- Extract results ---
    if [ -f "$run_dir/metrics.json" ]; then
        $PYTHON -c "
import json
m = json.load(open('$run_dir/metrics.json'))
dice = m.get('best_full_eval_dice', m.get('val_dice', 0))
tau = m.get('best_full_eval_tau_star', '?')
td = m.get('train_dice', '?')
vd = m.get('val_dice', '?')
print(f'${config_path},$run_id,{dice},?,{tau},{td},{vd},ok')
" >> "$RESULTS_FILE" 2>/dev/null || echo "$config_path,$run_id,?,?,?,?,?,metric_error" >> "$RESULTS_FILE"
    else
        echo "$config_path,$run_id,?,?,?,?,?,no_metrics" >> "$RESULTS_FILE"
    fi

    # --- GPU memory cleanup between runs ---
    echo "  🧹 Cleaning GPU memory..."
    $PYTHON -c "import torch, gc; gc.collect(); torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 10

    echo "  END: $(date)"
done

echo ""
echo "============================================================"
echo "  SWEEP COMPLETE — $(date)"
echo "  Results: $RESULTS_FILE"
echo "============================================================"
cat "$RESULTS_FILE"
