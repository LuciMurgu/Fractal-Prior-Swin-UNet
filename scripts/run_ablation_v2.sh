#!/bin/bash
# Run clean ablation v2 — one config at a time, sequential
# Each transition changes EXACTLY one variable
set -e

cd "/home/lucian/Codex/Swim -Fractal"
source .venv/bin/activate

CONFIGS=(
    "A_baseline"
    "B_lfd_gate"
    "C_lfd_spade1"
    "D_pde_spade2"
    "E_hessian"
    "F_skel_recall"
    "G_fractal_bce"
    "H_full"
)

echo "[$(date)] Starting clean ablation v2 (${#CONFIGS[@]} configs)"

for cfg in "${CONFIGS[@]}"; do
    run_id="ablation_v2_${cfg}"
    config_path="configs/ablation_v2/${cfg}.yaml"

    # Skip if already completed
    if [ -f "runs/${run_id}/metrics.json" ]; then
        dice=$(python3 -c "import json; m=json.load(open('runs/${run_id}/metrics.json')); print(f'{m.get(\"best_full_eval_dice\", 0):.4f}')")
        echo "[$(date)] SKIP ${cfg} — already done (dice=${dice})"
        continue
    fi

    echo "[$(date)] ▶ Starting ${cfg}"
    python -m fractal_swin_unet.train \
        --config "${config_path}" \
        --run_id "${run_id}" \
        --require_gpu \
        2>&1 | tee "runs/${run_id}.log" || true

    if [ -f "runs/${run_id}/metrics.json" ]; then
        dice=$(python3 -c "import json; m=json.load(open('runs/${run_id}/metrics.json')); print(f'{m.get(\"best_full_eval_dice\", 0):.4f}')")
        echo "[$(date)] ✓ ${cfg} completed — dice=${dice}"
    else
        echo "[$(date)] ✗ ${cfg} FAILED — no metrics.json"
    fi
done

echo ""
echo "[$(date)] ═══ ABLATION V2 RESULTS ═══"
echo "Config | Dice | Δ_vs_A | Change"
echo "-------|------|--------|-------"
for cfg in "${CONFIGS[@]}"; do
    run_id="ablation_v2_${cfg}"
    if [ -f "runs/${run_id}/metrics.json" ]; then
        python3 -c "
import json
m = json.load(open('runs/${run_id}/metrics.json'))
dice = m.get('best_full_eval_dice', 0)
print(f'${cfg} | {dice:.4f} | — | —')
"
    else
        echo "${cfg} | FAILED | — | —"
    fi
done
echo ""
echo "[$(date)] Done."
