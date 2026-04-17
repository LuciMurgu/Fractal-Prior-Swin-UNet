#!/usr/bin/env python3
"""Statistical analysis of cross-dataset results.

Computes:
  1. Per-image paired Wilcoxon signed-rank test (baseline vs fractal)
  2. Per-dataset summary with confidence intervals
  3. Effect size (Cohen's d)

Uses existing per-image JSON data — no GPU needed.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

from fractal_swin_unet.stats import (
    bootstrap_ci,
    cohens_d,
    paired_wilcoxon,
)

RESULTS_DIR = Path("reports/overnight_chain/phase2_cross_dataset")
OUTPUT_DIR = Path("reports/evidence_pack")

DATASETS = ["FIVES", "DRIVE", "CHASE_DB1", "STARE", "HRF"]

# Use tau_star from each model's result for fair comparison
# (each model gets its own optimal threshold)


def load_result(model: str, dataset: str) -> dict | None:
    f = RESULTS_DIR / f"{model}_on_{dataset}.json"
    if not f.exists():
        return None
    return json.load(open(f))


def per_image_metric(result: dict, metric_prefix: str) -> np.ndarray:
    """Extract per-image metric at the model's tau_star."""
    tau = result["tau_star"]
    key = f"{metric_prefix}_tau_{tau:.2f}"
    return np.array([img[key] for img in result["per_image"]])





def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  STATISTICAL ANALYSIS — Paired Wilcoxon Signed-Rank Test")
    print("  H₀: no difference between baseline and fractal")
    print("  H₁: fractal clDice > baseline clDice (one-sided)")
    print("=" * 80)

    all_stats = []

    for ds in DATASETS:
        baseline = load_result("baseline", ds)
        fractal = load_result("fractal", ds)

        if not baseline or not fractal:
            print(f"\n⚠️  SKIP {ds}: missing data")
            continue

        n_b = baseline["n_images"]
        n_f = fractal["n_images"]

        # Get per-image clDice at each model's own tau_star
        b_cldice = per_image_metric(baseline, "cldice")
        f_cldice = per_image_metric(fractal, "cldice")
        b_dice = per_image_metric(baseline, "dice")
        f_dice = per_image_metric(fractal, "dice")

        # Handle different n (use min)
        n = min(len(b_cldice), len(f_cldice))
        b_cldice = b_cldice[:n]
        f_cldice = f_cldice[:n]
        b_dice = b_dice[:n]
        f_dice = f_dice[:n]

        # Wilcoxon signed-rank test (one-sided: fractal > baseline)
        diff_cldice = f_cldice - b_cldice
        diff_dice = f_dice - b_dice

        # clDice test
        if np.all(diff_cldice == 0):
            w_cl, p_cl = 0.0, 1.0
        else:
            w_cl, p_cl_two = stats.wilcoxon(f_cldice, b_cldice, alternative="greater")
            p_cl = p_cl_two

        # Dice test
        if np.all(diff_dice == 0):
            w_d, p_d = 0.0, 1.0
        else:
            w_d, p_d_two = stats.wilcoxon(f_dice, b_dice, alternative="greater")
            p_d = p_d_two

        # Effect sizes
        d_cldice = cohens_d(f_cldice, b_cldice)
        d_dice = cohens_d(f_dice, b_dice)

        ci_cldice = bootstrap_ci(f_cldice, b_cldice, n_bootstrap=10000, seed=42)
        ci_dice = bootstrap_ci(f_dice, b_dice, n_bootstrap=10000, seed=42)

        sig_cl = "***" if p_cl < 0.001 else "**" if p_cl < 0.01 else "*" if p_cl < 0.05 else "ns"
        sig_d = "***" if p_d < 0.001 else "**" if p_d < 0.01 else "*" if p_d < 0.05 else "ns"

        print(f"\n{'─' * 70}")
        print(f"  {ds} (n={n} images)")
        print(f"{'─' * 70}")
        print(f"  clDice: ΔMean={diff_cldice.mean():+.4f} "
              f"95%CI=[{ci_cldice[0]:+.4f}, {ci_cldice[1]:+.4f}] "
              f"p={p_cl:.4f} {sig_cl}  d={d_cldice:.3f}")
        print(f"  Dice:   ΔMean={diff_dice.mean():+.4f} "
              f"95%CI=[{ci_dice[0]:+.4f}, {ci_dice[1]:+.4f}] "
              f"p={p_d:.4f} {sig_d}  d={d_dice:.3f}")

        stat = {
            "dataset": ds,
            "n": n,
            "b_cldice_mean": float(b_cldice.mean()),
            "b_cldice_std": float(b_cldice.std(ddof=1)) if n > 1 else 0.0,
            "f_cldice_mean": float(f_cldice.mean()),
            "f_cldice_std": float(f_cldice.std(ddof=1)) if n > 1 else 0.0,
            "b_dice_mean": float(b_dice.mean()),
            "b_dice_std": float(b_dice.std(ddof=1)) if n > 1 else 0.0,
            "f_dice_mean": float(f_dice.mean()),
            "f_dice_std": float(f_dice.std(ddof=1)) if n > 1 else 0.0,
            "cldice_delta_mean": float(diff_cldice.mean()),
            "cldice_delta_ci_lo": float(ci_cldice[0]),
            "cldice_delta_ci_hi": float(ci_cldice[1]),
            "cldice_p_value": float(p_cl),
            "cldice_significant": sig_cl,
            "cldice_cohens_d": float(d_cldice),
            "dice_delta_mean": float(diff_dice.mean()),
            "dice_delta_ci_lo": float(ci_dice[0]),
            "dice_delta_ci_hi": float(ci_dice[1]),
            "dice_p_value": float(p_d),
            "dice_significant": sig_d,
            "dice_cohens_d": float(d_dice),
        }
        all_stats.append(stat)

    # Save JSON
    with open(OUTPUT_DIR / "wilcoxon_results.json", "w") as f:
        json.dump(all_stats, f, indent=2)

    # Generate markdown table
    md_path = OUTPUT_DIR / "statistical_tests.md"
    with open(md_path, "w") as f:
        f.write("# Statistical Analysis — Paired Wilcoxon Signed-Rank Test\n\n")
        f.write("H₀: No difference between baseline and fractal clDice (per-image paired test)\n\n")
        f.write("| Dataset | n | Baseline (mean±std) | Fractal (mean±std) | ΔclDice | 95% CI | p-value | Sig | Cohen's d | ΔDice | p (Dice) |\n")
        f.write("|---------|---|--------------------|--------------------|---------|--------|---------|-----|-----------|-------|----------|\n")
        for s in all_stats:
            f.write(f"| {s['dataset']} | {s['n']} | "
                    f"{s['b_cldice_mean']:.4f}±{s['b_cldice_std']:.4f} | "
                    f"{s['f_cldice_mean']:.4f}±{s['f_cldice_std']:.4f} | "
                    f"{s['cldice_delta_mean']:+.4f} | "
                    f"[{s['cldice_delta_ci_lo']:+.4f}, {s['cldice_delta_ci_hi']:+.4f}] | "
                    f"{s['cldice_p_value']:.4f} | "
                    f"{s['cldice_significant']} | "
                    f"{s['cldice_cohens_d']:.3f} | "
                    f"{s['dice_delta_mean']:+.4f} | "
                    f"{s['dice_p_value']:.4f} |\n")

    print(f"\n📊 Results saved: {md_path}")
    print(f"📊 JSON saved: {OUTPUT_DIR / 'wilcoxon_results.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
