#!/usr/bin/env python3
"""Post-hoc clDice evaluation using the existing eval pipeline.

Runs `python -m fractal_swin_unet.eval` for each checkpoint and
extracts clDice from the threshold sweep output.
"""
import json
import subprocess
import sys
from pathlib import Path


# Runs to evaluate: (run_id, config_path)
RUNS = [
    # Ablation (HRF)
    ("ablation_v2_A_baseline", "configs/ablation/A_baseline.yaml"),
    ("ablation_v2_B_lfd_gate", "configs/ablation/B_lfd_gate.yaml"),
    ("ablation_v2_C_lfd_spade1", "configs/ablation/C_lfd_spade1.yaml"),
    ("ablation_v2_D_pde_spade2", "configs/ablation/D_pde_spade2.yaml"),
    ("ablation_v2_E_hessian", "configs/ablation/E_hessian.yaml"),
    ("ablation_v2_F_skel_recall", "configs/ablation/F_skel_recall.yaml"),
    ("ablation_v2_G_fractal_bce", "configs/ablation/G_fractal_bce.yaml"),
    ("ablation_v2_H_full", "configs/ablation/H_full_stack.yaml"),
    # OPT (FIVES)
    ("opt_A_pde15", "configs/optim/OPT_A_more_pde_steps.yaml"),
    ("opt_B_finer_lfd", "configs/optim/OPT_B_finer_lfd.yaml"),
    ("opt_C_spade64_skel", "configs/optim/OPT_C_bigger_spade_skel.yaml"),
    ("opt_D_longer", "configs/optim/OPT_D_longer_training.yaml"),
]


def run_eval(run_id: str, config_path: str) -> dict:
    """Run eval for a single checkpoint and extract metrics."""
    ckpt = Path(f"runs/{run_id}/checkpoint_best.pt")
    if not ckpt.exists():
        return {"error": "missing checkpoint"}
    if not Path(config_path).exists():
        return {"error": f"missing config: {config_path}"}

    cmd = [
        sys.executable, "-m", "fractal_swin_unet.eval",
        "--config", config_path,
        "--run_id", run_id,
    ]
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        output = result.stdout + result.stderr
        
        # Try to read updated metrics.json
        metrics_path = Path(f"runs/{run_id}/metrics.json")
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            return metrics
        
        return {"output": output[-500:], "returncode": result.returncode}
    except Exception as e:
        return {"error": str(e)}


def main():
    print(f"{'Run':<35} {'BestDice':>10} {'clDice':>10} {'tau*':>6}")
    print("=" * 65)
    
    for run_id, config_path in RUNS:
        metrics = run_eval(run_id, config_path)
        
        if "error" in metrics:
            print(f"{run_id:<35} ERROR: {metrics['error']}")
            continue
        
        dice = metrics.get("best_dice", metrics.get("dice", "N/A"))
        cldice = metrics.get("cldice_at_tau_star", "N/A")
        tau = metrics.get("tau_star", "N/A")
        
        if isinstance(dice, float):
            dice = f"{dice:.4f}"
        if isinstance(cldice, float):
            cldice = f"{cldice:.4f}"
        if isinstance(tau, float):
            tau = f"{tau:.2f}"
            
        print(f"{run_id:<35} {dice:>10} {cldice:>10} {tau:>6}")


if __name__ == "__main__":
    main()
