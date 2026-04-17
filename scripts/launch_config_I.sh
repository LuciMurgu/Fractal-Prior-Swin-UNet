#!/bin/bash
# Launch Config I after Config H finishes
set -e

cd "/home/lucian/Codex/Swim -Fractal"
source .venv/bin/activate

echo "[$(date)] Waiting for Config H to finish..."
while pgrep -f "ablation_H_full_stack" > /dev/null 2>&1; do
    sleep 30
done
echo "[$(date)] Config H finished. Launching Config I..."

python -m fractal_swin_unet.train \
    --config configs/ablation/I_rot_equivariant_full.yaml \
    --run_id ablation_I_rot_equivariant_full \
    --require_gpu \
    2>&1 | tee runs/ablation_I_rot_equivariant_full.log

echo "[$(date)] Config I finished."
