#!/usr/bin/env bash
# ============================================================
# Full Metrics Evaluation — 3 Key Models
# Computes: Dice, clDice, AUC, Se, Sp, Acc, F1 on val+test
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

PYTHON=".venv/bin/python3"

echo "============================================================"
echo "  FULL METRICS EVALUATION — $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================================"

# ── 1. OPT-D (Fractal, best model — Dice champion) ─────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  [1/3] OPT-D (Fractal Prior — BEST Dice)"
echo "  Checkpoint: runs/opt_D_longer/checkpoint_best.pt"
echo "  Config: configs/optim/OPT_D_longer_training.yaml"
echo "═══════════════════════════════════════════════════════════"
$PYTHON -m fractal_swin_unet.eval \
    --config configs/optim/OPT_D_longer_training.yaml \
    --checkpoint runs/opt_D_longer/checkpoint_best.pt \
    --run_id eval_optD_full_metrics

echo ""
echo "  ✅ OPT-D eval complete"

# GPU cleanup
$PYTHON -c "import torch, gc; gc.collect(); torch.cuda.empty_cache()" 2>/dev/null || true
sleep 5

# ── 2. Baseline 200ep (Control — no fractal prior) ──────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  [2/3] Baseline 200ep (NO fractal prior — control)"
echo "  Checkpoint: runs/baseline_200ep/checkpoint_best.pt"  
echo "  Config: configs/optim/BASELINE_200ep.yaml"
echo "═══════════════════════════════════════════════════════════"
$PYTHON -m fractal_swin_unet.eval \
    --config configs/optim/BASELINE_200ep.yaml \
    --checkpoint runs/baseline_200ep/checkpoint_best.pt \
    --run_id eval_baseline_full_metrics

echo ""
echo "  ✅ Baseline eval complete"

# GPU cleanup
$PYTHON -c "import torch, gc; gc.collect(); torch.cuda.empty_cache()" 2>/dev/null || true
sleep 5

# ── 3. Config F (Skeleton recall — clDice champion) ──────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  [3/3] Config F (Fractal + Skeleton Recall — BEST clDice)"
echo "  Checkpoint: runs/xdataset_fives_configF/checkpoint_best.pt"
echo "  Config: configs/cross_dataset/fives_configF.yaml"
echo "═══════════════════════════════════════════════════════════"
$PYTHON -m fractal_swin_unet.eval \
    --config configs/cross_dataset/fives_configF.yaml \
    --checkpoint runs/xdataset_fives_configF/checkpoint_best.pt \
    --run_id eval_configF_full_metrics

echo ""
echo "  ✅ Config F eval complete"

# ── Summary ──────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  ALL EVALUATIONS COMPLETE — $(date)"
echo "============================================================"
echo ""

for run_id in eval_optD_full_metrics eval_baseline_full_metrics eval_configF_full_metrics; do
    mf="runs/$run_id/metrics.json"
    if [ -f "$mf" ]; then
        echo "─── $run_id ───"
        $PYTHON -c "
import json
m = json.load(open('$mf'))
# Val metrics
print(f'  VAL:  Dice={m.get(\"best_dice\",\"?\"):.4f}  clDice={m.get(\"best_cldice\",\"?\"):.4f}  AUC={m.get(\"auc_roc\",\"?\"):.4f}  Se={m.get(\"se_tau_star\",\"?\"):.4f}  Sp={m.get(\"sp_tau_star\",\"?\"):.4f}  Acc={m.get(\"acc_tau_star\",\"?\"):.4f}  F1={m.get(\"f1_tau_star\",\"?\"):.4f}  tau*={m.get(\"tau_star\",\"?\")}')
# Test metrics
td = m.get('test_dice_tau_star')
if td is not None:
    print(f'  TEST: Dice={td:.4f}  clDice={m.get(\"test_cldice_tau_star\",\"?\"):.4f}  AUC={m.get(\"test_auc_roc\",\"?\"):.4f}  Se={m.get(\"test_se_tau_star\",\"?\"):.4f}  Sp={m.get(\"test_sp_tau_star\",\"?\"):.4f}  Acc={m.get(\"test_acc_tau_star\",\"?\"):.4f}  F1={m.get(\"test_f1_tau_star\",\"?\"):.4f}')
"
        echo ""
    else
        echo "─── $run_id ─── NO METRICS"
    fi
done

echo "DONE! 🎉"
