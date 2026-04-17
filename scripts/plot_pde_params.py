#!/usr/bin/env python3
"""Plot PDE parameter evolution from pde_params.jsonl.

Produces reports/journal_figures/pde_params_evolution.png with one subplot
per parameter showing how the learned values drift during training.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent.parent

# Search for pde_params.jsonl in all runs
SEARCH_DIRS = [
    PROJECT / "runs" / "opt_D_v2_200ep",
    PROJECT / "runs" / "opt_F_lacunarity",
    PROJECT / "runs" / "opt_G_percolation",
]

PARAM_NAMES = ["alpha", "lambda", "sigma", "beta", "xi", "eta", "nu", "gamma", "omega"]
PARAM_LABELS = ["α (diffusion)", "λ (fidelity)", "σ (blur)", "β (conductance)",
                "ξ (threshold num)", "η (threshold den)", "ν (curvature)",
                "γ (curvature offset)", "ω (fractal mod)"]

OUT_DIR = PROJECT / "reports" / "journal_figures"


def load_jsonl(path: Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all available pde_params.jsonl files
    all_runs = {}
    for d in SEARCH_DIRS:
        jsonl = d / "pde_params.jsonl"
        if jsonl.exists():
            entries = load_jsonl(jsonl)
            if entries:
                all_runs[d.name] = entries
                print(f"  ✅ {d.name}: {len(entries)} epochs")

    if not all_runs:
        print("  ⚠️  No pde_params.jsonl found in any run directory.")
        print("     Run experiments with PDE logging enabled first.")
        return

    # Plot
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle("Learned PDE Parameter Evolution", fontsize=14, fontweight="bold")

    colors = {"opt_D_v2_200ep": "#2196F3", "opt_F_lacunarity": "#4CAF50",
              "opt_G_percolation": "#FF5722"}

    for idx, (param, label) in enumerate(zip(PARAM_NAMES, PARAM_LABELS)):
        ax = axes[idx // 3][idx % 3]
        for run_name, entries in all_runs.items():
            epochs = [e["epoch"] for e in entries]
            values = [e.get(param, 0) for e in entries]
            color = colors.get(run_name, "#666")
            ax.plot(epochs, values, label=run_name, color=color, linewidth=1.5)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7)

    plt.tight_layout()
    out_path = OUT_DIR / "pde_params_evolution.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✅ PDE parameter plot saved to {out_path}")


if __name__ == "__main__":
    main()
