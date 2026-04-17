#!/usr/bin/env python3
"""Compile journal comparison table from run metrics.

Reads metrics.json from each completed run and produces:
- reports/journal_table.md  (markdown table)
- reports/journal_table.csv (machine-readable)
"""
import json
import csv
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT / "runs"
OUT_DIR = PROJECT / "reports"

# Runs to include in the journal table
RUNS = [
    ("baseline_200ep", "Baseline (200ep)"),
    ("opt_D_v2_200ep", "OPT-D v2 (3-ch PDE)"),
    ("opt_F_lacunarity", "OPT-F (+Lac, 4-ch)"),
    ("opt_G_percolation", "OPT-G (+Lac+Perc, 5-ch)"),
]

METRIC_KEYS = [
    ("best_dice", "Dice↑"),
    ("best_cldice", "clDice↑"),
    ("auc_roc", "AUC↑"),
    ("se_tau_star", "Se↑"),
    ("sp_tau_star", "Sp↑"),
    ("acc_tau_star", "Acc↑"),
    ("f1_tau_star", "F1↑"),
    ("tau_star", "τ*"),
]

# Also check test metrics
TEST_METRIC_KEYS = [
    ("test_dice_tau_star", "Test Dice"),
    ("test_cldice_tau_star", "Test clDice"),
    ("test_auc_roc", "Test AUC"),
    ("test_se_tau_star", "Test Se"),
    ("test_sp_tau_star", "Test Sp"),
    ("test_acc_tau_star", "Test Acc"),
    ("test_f1_tau_star", "Test F1"),
]


def _find_metrics(run_id: str) -> dict | None:
    """Find metrics.json in run dir or eval sub-dir."""
    candidates = [
        RUNS_DIR / run_id / "metrics.json",
        RUNS_DIR / f"{run_id}_eval" / "metrics.json",
    ]
    for p in candidates:
        if p.exists():
            return json.loads(p.read_text())
    return None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for run_id, label in RUNS:
        m = _find_metrics(run_id)
        if m is None:
            print(f"  ⚠️  {label} ({run_id}): no metrics.json found")
            rows.append({"run_id": run_id, "label": label})
            continue

        row = {"run_id": run_id, "label": label}
        for key, _ in METRIC_KEYS + TEST_METRIC_KEYS:
            row[key] = m.get(key)
        rows.append(row)
        print(f"  ✅ {label}: Dice={row.get('best_dice', '—')}")

    # Write CSV
    csv_path = OUT_DIR / "journal_table.csv"
    all_keys = ["run_id", "label"] + [k for k, _ in METRIC_KEYS + TEST_METRIC_KEYS]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        w.writerows(rows)

    # Write MD
    md_path = OUT_DIR / "journal_table.md"
    with open(md_path, "w") as f:
        f.write("# Journal Comparison Table\n\n")
        f.write("> Generated from completed experiment runs.\n\n")

        # Val metrics table
        f.write("## Validation Metrics (at τ*)\n\n")
        header = "| Configuration |" + "|".join(h for _, h in METRIC_KEYS) + "|\n"
        sep = "|" + "|".join(["---"] * (len(METRIC_KEYS) + 1)) + "|\n"
        f.write(header)
        f.write(sep)
        for r in rows:
            cells = [r["label"]]
            for key, _ in METRIC_KEYS:
                v = r.get(key)
                cells.append(f"{v:.4f}" if v is not None else "—")
            f.write("| " + " | ".join(cells) + " |\n")

        # Test metrics table
        f.write("\n## Test Metrics (at τ* from val)\n\n")
        header = "| Configuration |" + "|".join(h for _, h in TEST_METRIC_KEYS) + "|\n"
        sep = "|" + "|".join(["---"] * (len(TEST_METRIC_KEYS) + 1)) + "|\n"
        f.write(header)
        f.write(sep)
        for r in rows:
            cells = [r["label"]]
            for key, _ in TEST_METRIC_KEYS:
                v = r.get(key)
                cells.append(f"{v:.4f}" if v is not None else "—")
            f.write("| " + " | ".join(cells) + " |\n")

        # Summary
        f.write("\n## Notes\n\n")
        f.write("- All runs use FIVES dataset, same train/val/test split (seed=123)\n")
        f.write("- τ* selected on validation set only (no test leakage)\n")
        f.write("- clDice computed with degenerate-case fix (commit 33e135c)\n")
        f.write("- Lacunarity uses vectorized computation without per-image normalization\n")
        f.write("- Percolation uses true edge-to-edge spanning condition\n")

    print(f"\n✅ Journal table written to {md_path}")
    print(f"   CSV at {csv_path}")


if __name__ == "__main__":
    main()
