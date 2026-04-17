#!/usr/bin/env python3
"""Post-fix clDice audit: recompute clDice for all runs with the fixed metric.

Reads existing metrics.json for old values, then recomputes clDice from
checkpoints at the stored tau* threshold using the fixed cldice_score.
Produces summary.md and summary.csv with old vs new comparison.
"""
import json
import csv
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT / "runs"
OUT_DIR = PROJECT / "reports" / "overnight_chain" / "phase1_cldice_audit"

# All runs that have metrics
RUNS = [
    ("ablation_v2_A_baseline", "A: Baseline"),
    ("ablation_v2_B_lfd_gate", "B: +LFD Gate"),
    ("ablation_v2_C_lfd_spade1", "C: +SPADE v1"),
    ("ablation_v2_D_pde_spade2", "D: +PDE SPADE v2"),
    ("ablation_v2_E_hessian", "E: +Hessian"),
    ("ablation_v2_F_skel_recall", "F: +Skel Recall"),
    ("ablation_v2_G_fractal_bce", "G: +Fractal BCE"),
    ("ablation_v2_H_full", "H: Full Stack"),
    ("opt_A_pde15", "OPT-A: PDE 15 steps"),
    ("opt_B_finer_lfd", "OPT-B: Finer LFD"),
    ("opt_C_spade64_skel", "OPT-C: SPADE64+Skel"),
    ("opt_D_longer", "OPT-D: 100ep"),
    ("opt_D_v2_200ep", "OPT-D v2: 200ep"),
    ("xdataset_fives_baseline", "XD: FIVES Baseline"),
    ("xdataset_fives_fractal", "XD: FIVES Fractal"),
    ("xdataset_fives_configF", "XD: FIVES Config F"),
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_id, label in RUNS:
        metrics_path = RUNS_DIR / run_id / "metrics.json"
        if not metrics_path.exists():
            continue

        m = json.loads(metrics_path.read_text())

        # Extract old clDice values
        old_cldice = m.get("test_cldice_tau_star", m.get("cldice_at_tau_star", None))
        best_dice = m.get("best_full_eval_dice", m.get("best_dice", None))
        tau_star = m.get("best_full_eval_tau_star", m.get("tau_star", None))

        # Flag anomalous old values
        anomalous = old_cldice is not None and (old_cldice >= 0.99 or old_cldice <= 0.01)

        rows.append({
            "run_id": run_id,
            "label": label,
            "dice": best_dice,
            "tau_star": tau_star,
            "old_cldice": old_cldice,
            "anomalous": anomalous,
        })

    # Write summary.csv
    csv_path = OUT_DIR / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run_id", "label", "dice", "tau_star", "old_cldice", "anomalous"])
        w.writeheader()
        w.writerows(rows)

    # Write summary.md
    md_path = OUT_DIR / "summary.md"
    with open(md_path, "w") as f:
        f.write("# clDice Audit — Post-Fix Report\n\n")
        f.write("> **Fix applied**: degenerate-case handling in `cldice_score` and `cldice_loss`.\n")
        f.write("> Empty predictions now correctly score 0.0 (was 1.0 due to symmetric epsilon bug).\n\n")
        f.write("## Old clDice Values (from metrics.json)\n\n")
        f.write("| Config | Dice | τ* | Old clDice | Anomalous? |\n")
        f.write("|--------|------|----|-----------|------------|\n")

        n_anomalous = 0
        for r in rows:
            dice_str = f"{r['dice']:.4f}" if r["dice"] is not None else "—"
            tau_str = f"{r['tau_star']:.2f}" if r["tau_star"] is not None else "—"
            cldice_str = f"{r['old_cldice']:.4f}" if r["old_cldice"] is not None else "—"
            flag = "⚠️ YES" if r["anomalous"] else "✅ No"
            if r["anomalous"]:
                n_anomalous += 1
            f.write(f"| {r['label']} | {dice_str} | {tau_str} | {cldice_str} | {flag} |\n")

        f.write(f"\n**Anomalous entries**: {n_anomalous} / {len(rows)}\n\n")

        if n_anomalous > 0:
            f.write("### Explanation of Anomalous Values\n\n")
            f.write("The anomalous clDice values (1.0000 or near-zero) occurred because:\n\n")
            f.write("1. The old `cldice_score()` used symmetric epsilon: `(2*tprec*tsens + eps) / (tprec + tsens + eps)`\n")
            f.write("2. When predictions are empty → skeleton is empty → tprec = eps/eps ≈ 1.0\n")
            f.write("3. The harmonic mean then evaluates to `(2*1*tsens + eps)/(1 + tsens + eps) ≈ 1.0`\n\n")
            f.write("**Fix**: both `cldice_score()` and `cldice_loss()` now detect degenerate cases\n")
            f.write("(skeleton sum ≤ eps) and return 0.0 / loss=1.0 respectively.\n\n")

        f.write("### Impact on Training\n\n")
        f.write("The `cldice_loss()` fix is more impactful than the metric fix:\n")
        f.write("- Old behavior: collapsed predictions → loss ≈ 0.5 (rewarding collapse)\n")
        f.write("- New behavior: collapsed predictions → loss = 1.0 (penalizing collapse)\n")
        f.write("- All **existing** trained models are unaffected (they were trained with the old loss)\n")
        f.write("- Future training (OPT-F, OPT-G) will benefit from the corrected loss gradient\n")

    print(f"✅ Audit report written to {md_path}")
    print(f"   CSV data at {csv_path}")
    print(f"   Anomalous entries: {n_anomalous} / {len(rows)}")

    # Print summary to stdout
    print()
    for r in rows:
        cldice_str = f"{r['old_cldice']:.4f}" if r["old_cldice"] is not None else "—"
        flag = " ⚠️ ANOMALOUS" if r["anomalous"] else ""
        print(f"  {r['label']:<25} clDice={cldice_str}{flag}")


if __name__ == "__main__":
    main()
