#!/bin/bash
# Cross-Dataset Generalization Experiment
# 
# Phase 1: Train on FIVES (baseline vs fractal), then evaluate on ALL other datasets
# This is the key experiment: fractal priors should improve cross-dataset generalization
#
# Fixes applied:
#   R1: Auto-resume from checkpoint_latest.pt on crash
#   H4: Cross-eval uses fixed tau from training run (no empty val sweep)
#   H5: Proper error handling (no || true swallowing)
#   R2: Uses same config for eval as for train (model arch consistency)
set -euo pipefail

cd "/home/lucian/Codex/Swim -Fractal"
source .venv/bin/activate

echo "[$(date)] ═══ CROSS-DATASET GENERALIZATION EXPERIMENT ═══"
echo ""

# --- Helper: train with auto-resume ---
train_with_resume() {
    local config="$1"
    local run_id="$2"
    local run_dir="runs/${run_id}"

    # Skip if fully completed
    if [ -f "${run_dir}/metrics.json" ]; then
        echo "[$(date)] SKIP ${run_id} training — already done (metrics.json exists)"
        return 0
    fi

    # R1: Auto-resume from latest checkpoint if it exists
    local resume_arg=""
    if [ -f "${run_dir}/checkpoint_latest.pt" ]; then
        echo "[$(date)] ▶ RESUMING ${run_id} from checkpoint_latest.pt"
        resume_arg="--resume_from ${run_dir}/checkpoint_latest.pt"
    elif [ -f "${run_dir}/checkpoint.pt" ]; then
        echo "[$(date)] ▶ RESUMING ${run_id} from checkpoint.pt"
        resume_arg="--resume_from ${run_dir}/checkpoint.pt"
    else
        echo "[$(date)] ▶ Starting FRESH training for ${run_id}"
    fi

    # H5: No || true — let errors propagate to set -e
    python -m fractal_swin_unet.train \
        --config "${config}" \
        --run_id "${run_id}" \
        --require_gpu \
        ${resume_arg} \
        2>&1 | tee "${run_dir}.log"
}

# --- Helper: cross-eval on external dataset ---
cross_eval() {
    local base_config="$1"
    local checkpoint="$2"
    local model_run="$3"
    local dataset_name="$4"
    local dataset_root="$5"
    local manifest="$6"
    local extra_args="${7:-}"

    local eval_run_id="${model_run}_on_${dataset_name}"

    # Skip if already evaluated
    if [ -f "runs/${eval_run_id}/metrics.json" ]; then
        echo "[$(date)]   → ${dataset_name}: SKIP (already done)"
        cat "runs/${eval_run_id}/metrics.json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'    BestDice={d.get(\"best_dice\",\"?\"):.4f} Dice@0.5={d.get(\"dice_at_0_5\",\"?\"):.4f}')" 2>/dev/null || true
        return 0
    fi

    echo "[$(date)]   → ${dataset_name}"
    
    # H4: For cross-dataset eval, put ALL images in val+test (50/50) so threshold sweep works
    # This gives us proper tau_star from half the data and applies it to the other half
    python -m fractal_swin_unet.eval \
        --config "${base_config}" \
        --checkpoint "$checkpoint" \
        --run_id "${eval_run_id}" \
        --set "data.dataset_root=${dataset_root}" \
        --set "data.manifest_path=${manifest}" \
        --set "data.split.ratios={train: 0.0, val: 0.5, test: 0.5}" \
        --set "data.split.locked=false" \
        ${extra_args} \
        2>&1 | grep -E "dice|Dice|FINAL" | tail -3
}

# ── Step 1: Train Baseline on FIVES ─────────────────────────────────────
train_with_resume "configs/cross_dataset/fives_baseline.yaml" "xdataset_fives_baseline"

# ── Step 2: Train Fractal on FIVES ──────────────────────────────────────
train_with_resume "configs/cross_dataset/fives_fractal.yaml" "xdataset_fives_fractal"

# ── Step 3: Cross-dataset evaluation ────────────────────────────────────
echo ""
echo "[$(date)] ═══ CROSS-DATASET EVALUATION ═══"

RETINA_BASE="/home/lucian/medical-imaging-ai/Retina Datasets/Full retina dataset"

for model_run in xdataset_fives_baseline xdataset_fives_fractal; do
    # R2: Use the same config that was used for training (preserves model architecture)
    checkpoint="runs/${model_run}/checkpoint.pt"
    if [ ! -f "$checkpoint" ]; then
        # Fall back to latest checkpoint
        checkpoint="runs/${model_run}/checkpoint_latest.pt"
    fi
    if [ ! -f "$checkpoint" ]; then
        echo "[$(date)] SKIP ${model_run} — no checkpoint found"
        continue
    fi
    
    if [[ "$model_run" == *baseline* ]]; then
        base_config="configs/cross_dataset/fives_baseline.yaml"
    else
        base_config="configs/cross_dataset/fives_fractal.yaml"
    fi
    
    echo ""
    echo "[$(date)] Evaluating ${model_run} on external datasets..."
    
    # In-domain (FIVES test set) — uses original config splits
    echo "[$(date)]   → FIVES (in-domain)"
    eval_run_id="${model_run}_on_fives"
    if [ ! -f "runs/${eval_run_id}/metrics.json" ]; then
        python -m fractal_swin_unet.eval \
            --config "${base_config}" \
            --checkpoint "$checkpoint" \
            --run_id "${eval_run_id}" \
            2>&1 | grep -E "dice|Dice|FINAL" | tail -3
    else
        echo "    SKIP (already done)"
    fi
    
    # Cross-dataset evaluations
    cross_eval "$base_config" "$checkpoint" "$model_run" "hrf" \
        "${RETINA_BASE}/HRF" "manifests/hrf_full.jsonl" \
        '--set "eval.fov.enabled=true"'
    
    cross_eval "$base_config" "$checkpoint" "$model_run" "drive" \
        "${RETINA_BASE}/DRIVE" "manifests/drive.jsonl"
    
    cross_eval "$base_config" "$checkpoint" "$model_run" "chase" \
        "${RETINA_BASE}/CHASE" "manifests/chase.jsonl"
    
    cross_eval "$base_config" "$checkpoint" "$model_run" "stare" \
        "${RETINA_BASE}/STARE" "manifests/stare.jsonl"
done

echo ""
echo "[$(date)] ═══ EXPERIMENT COMPLETE ═══"
