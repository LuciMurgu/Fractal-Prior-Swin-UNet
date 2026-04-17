"""AutoResearch: Autonomous hyperparameter optimization loop.

Inspired by Andrej Karpathy's autoresearch concept.
Each iteration:
  1. Mutate 1-2 hyperparameters from the current best config
  2. Run a short training (20 epochs)
  3. Compare best_full_eval_dice against current champion
  4. Keep improvement or revert — the ratchet only goes up
  5. Log everything to leaderboard

Usage:
    python -m autoresearch.runner --max_experiments 20 --epochs_per_run 20
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [AutoResearch] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("autoresearch")

# ── Search Space ─────────────────────────────────────────────────────────

SEARCH_SPACE: dict[str, list[Any]] = {
    # Architecture
    "model.embed_dim": [32, 48, 64],
    "model.depths": [[1, 1, 1], [2, 1, 1], [2, 2, 1]],
    "model.fractal_diffusion_spade.hidden_dim": [16, 32, 48],
    "model.fractal_diffusion_spade.n_steps": [3, 5, 7, 10],
    # Training
    "train.lr": [2e-4, 3e-4, 5e-4, 8e-4, 1e-3],
    "train.weight_decay": [0.001, 0.005, 0.01, 0.02],
    "train.steps_per_epoch": [220, 300, 400],
    "train.grad_clip_norm": [0.5, 1.0, 2.0],
    # Loss weights
    "loss.skeleton_recall.weight": [0.05, 0.10, 0.15, 0.20, 0.30],
    "loss.fractal_bce.weight": [0.05, 0.10, 0.15, 0.20],
    "loss.fractal_bce.alpha": [0.5, 1.0, 1.5, 2.0, 3.0],
    "loss.cldice.weight": [0.05, 0.10, 0.12, 0.15, 0.20],
    "loss.focal.weight": [0.05, 0.10, 0.15, 0.20],
    # Data
    "data.patch_size": [[256, 256], [384, 384], [512, 512]],
    "data.patch_sampling.p_vessel": [0.6, 0.7, 0.8],
    "data.photometric_aug.brightness": [0.05, 0.08, 0.12],
    "data.photometric_aug.contrast": [0.10, 0.15, 0.20, 0.25],
}

# Params that are unsafe to change together (can cause OOM or instability)
CONFLICT_GROUPS = [
    {"model.embed_dim", "model.depths"},  # Both increase VRAM
    {"data.patch_size", "model.embed_dim"},  # Both increase VRAM
]


def _set_nested(d: dict, dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using dotted key notation."""
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _get_nested(d: dict, dotted_key: str, default: Any = None) -> Any:
    """Get a value from a nested dict using dotted key notation."""
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.get(k, {})
        if not isinstance(d, dict):
            return default
    return d.get(keys[-1], default)


def propose_mutation(
    base_config: dict,
    n_mutations: int = 1,
    rng: random.Random | None = None,
) -> tuple[dict, list[str]]:
    """Propose a mutated config by changing n_mutations hyperparameters.

    Returns:
        (mutated_config, list of mutation descriptions)
    """
    if rng is None:
        rng = random.Random()

    config = copy.deepcopy(base_config)
    candidates = list(SEARCH_SPACE.keys())
    rng.shuffle(candidates)

    mutations: list[str] = []
    changed_keys: set[str] = set()

    for key in candidates:
        if len(mutations) >= n_mutations:
            break

        # Skip if conflicts with already-changed params
        skip = False
        for group in CONFLICT_GROUPS:
            if key in group and group & changed_keys:
                skip = True
                break
        if skip:
            continue

        current_value = _get_nested(config, key)
        options = [v for v in SEARCH_SPACE[key] if v != current_value]
        if not options:
            continue

        new_value = rng.choice(options)
        _set_nested(config, key, new_value)
        mutations.append(f"{key}: {current_value} → {new_value}")
        changed_keys.add(key)

    return config, mutations


# ── Experiment Runner ────────────────────────────────────────────────────

def run_experiment(
    config: dict,
    run_id: str,
    project_dir: Path,
    epochs: int = 20,
    timeout_sec: int = 3600,
) -> dict | None:
    """Run a training experiment and return the metrics.

    Returns:
        metrics dict if successful, None if failed.
    """
    # Write config to temp file
    config_path = project_dir / "autoresearch" / f"{run_id}.yaml"
    config_path.parent.mkdir(exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Build command
    cmd = [
        sys.executable, "-m", "fractal_swin_unet.train",
        "--config", str(config_path),
        "--run_id", run_id,
        "--require_gpu",
    ]

    log.info(f"▶ Starting experiment: {run_id}")
    log.info(f"  Config: {config_path}")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        log.warning(f"✗ Experiment {run_id} timed out after {timeout_sec}s")
        return None

    if result.returncode != 0:
        # Try to extract error
        stderr_tail = result.stderr[-500:] if result.stderr else "no stderr"
        stdout_tail = result.stdout[-500:] if result.stdout else "no stdout"
        log.warning(f"✗ Experiment {run_id} failed (exit {result.returncode})")
        log.warning(f"  stderr: {stderr_tail}")
        log.warning(f"  stdout: {stdout_tail}")
        return None

    # Read metrics
    metrics_path = project_dir / "runs" / run_id / "metrics.json"
    if not metrics_path.exists():
        log.warning(f"✗ No metrics.json found for {run_id}")
        return None

    with open(metrics_path) as f:
        metrics = json.load(f)

    dice = metrics.get("best_full_eval_dice", 0.0)
    epoch = metrics.get("best_full_eval_epoch", "?")
    tau = metrics.get("best_full_eval_tau_star", "?")
    log.info(f"✓ Experiment {run_id}: dice={dice:.4f} (epoch {epoch}, τ*={tau})")

    return metrics


# ── Leaderboard ──────────────────────────────────────────────────────────

class Leaderboard:
    """Persistent leaderboard tracking all experiments."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: list[dict] = []
        if path.exists():
            with open(path) as f:
                self.entries = json.load(f)

    def add(
        self,
        run_id: str,
        dice: float,
        mutations: list[str],
        config: dict,
        is_improvement: bool,
    ) -> None:
        self.entries.append({
            "run_id": run_id,
            "dice": dice,
            "mutations": mutations,
            "is_improvement": is_improvement,
            "timestamp": datetime.now().isoformat(),
            "config_snapshot": {
                k: _get_nested(config, k)
                for k in SEARCH_SPACE.keys()
            },
        })
        self._save()

    def _save(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self.entries, f, indent=2, default=str)

    @property
    def best_dice(self) -> float:
        if not self.entries:
            return 0.0
        return max(e["dice"] for e in self.entries)

    def summary(self) -> str:
        if not self.entries:
            return "No experiments yet."
        lines = ["# AutoResearch Leaderboard", ""]
        lines.append(f"Total experiments: {len(self.entries)}")
        improvements = [e for e in self.entries if e["is_improvement"]]
        lines.append(f"Improvements found: {len(improvements)}")
        lines.append(f"Best dice: {self.best_dice:.4f}")
        lines.append("")
        lines.append("| # | Run ID | Dice | Improvement | Mutations |")
        lines.append("|---|--------|------|-------------|-----------|")
        for i, e in enumerate(self.entries):
            mark = "✅" if e["is_improvement"] else "❌"
            muts = "; ".join(e["mutations"][:2])
            lines.append(f"| {i+1} | {e['run_id']} | {e['dice']:.4f} | {mark} | {muts} |")
        return "\n".join(lines)


# ── Main Loop ────────────────────────────────────────────────────────────

def load_base_config(path: Path) -> dict:
    """Load the baseline config YAML."""
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoResearch: autonomous HPO loop")
    parser.add_argument("--max_experiments", type=int, default=20,
                        help="Maximum number of experiments to run")
    parser.add_argument("--epochs_per_run", type=int, default=20,
                        help="Training epochs per experiment")
    parser.add_argument("--base_config", type=str,
                        default="configs/ablation/H_full_stack.yaml",
                        help="Path to the baseline config")
    parser.add_argument("--baseline_dice", type=float, default=0.8096,
                        help="Current best dice score to beat")
    parser.add_argument("--n_mutations", type=int, default=1,
                        help="Number of params to mutate per experiment (1-3)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Max seconds per experiment")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent.parent
    rng = random.Random(args.seed)

    log.info("=" * 60)
    log.info("AutoResearch: Fractal-Prior Vessel Segmentation")
    log.info("=" * 60)
    log.info(f"Base config: {args.base_config}")
    log.info(f"Baseline dice: {args.baseline_dice:.4f}")
    log.info(f"Max experiments: {args.max_experiments}")
    log.info(f"Epochs per run: {args.epochs_per_run}")
    log.info(f"Mutations per run: {args.n_mutations}")
    log.info("")

    # Load base config
    base_config = load_base_config(project_dir / args.base_config)

    # Override epochs for short runs
    base_config["train"]["epochs"] = args.epochs_per_run
    # Ensure full eval happens: at halfway and at the end
    eval_interval = max(1, args.epochs_per_run // 2)
    base_config["train"]["full_eval_every_epochs"] = eval_interval
    base_config["train"]["save_every_epochs"] = args.epochs_per_run

    # Initialize leaderboard
    lb_path = project_dir / "autoresearch" / "leaderboard.json"
    leaderboard = Leaderboard(lb_path)

    # Track current champion
    champion_config = copy.deepcopy(base_config)
    champion_dice = args.baseline_dice

    # Add baseline entry
    if not leaderboard.entries:
        leaderboard.add(
            "baseline_H", champion_dice,
            ["Config H baseline"], champion_config, True,
        )

    n_improvements = 0
    n_failures = 0
    start_time = time.time()

    for i in range(args.max_experiments):
        elapsed = time.time() - start_time
        log.info(f"─── Experiment {i+1}/{args.max_experiments} "
                 f"(elapsed: {elapsed/3600:.1f}h, champion: {champion_dice:.4f}) ───")

        # Propose mutation
        n_mut = rng.choice([1, 1, 1, 2]) if args.n_mutations <= 2 else args.n_mutations
        candidate_config, mutations = propose_mutation(champion_config, n_mutations=n_mut, rng=rng)

        if not mutations:
            log.warning("No valid mutations found, skipping")
            continue

        for m in mutations:
            log.info(f"  Mutation: {m}")

        run_id = f"autoresearch_{i:03d}"

        # Run experiment
        metrics = run_experiment(
            candidate_config, run_id, project_dir,
            epochs=args.epochs_per_run,
            timeout_sec=args.timeout,
        )

        if metrics is None:
            n_failures += 1
            leaderboard.add(run_id, 0.0, mutations, candidate_config, False)
            log.info(f"  Result: FAILED (failures so far: {n_failures})")
            continue

        dice = metrics.get("best_full_eval_dice", 0.0)
        is_improvement = dice > champion_dice

        leaderboard.add(run_id, dice, mutations, candidate_config, is_improvement)

        if is_improvement:
            n_improvements += 1
            delta = dice - champion_dice
            log.info(f"  🎉 NEW CHAMPION! {dice:.4f} (+{delta:.4f})")
            log.info(f"  Improvements so far: {n_improvements}")
            champion_dice = dice
            champion_config = copy.deepcopy(candidate_config)

            # Save champion config
            champion_path = project_dir / "autoresearch" / "champion.yaml"
            with open(champion_path, "w") as f:
                yaml.dump(champion_config, f, default_flow_style=False, sort_keys=False)
            log.info(f"  Saved champion config to {champion_path}")
        else:
            delta = dice - champion_dice
            log.info(f"  No improvement: {dice:.4f} ({delta:+.4f})")

        # Save leaderboard summary
        summary_path = project_dir / "autoresearch" / "leaderboard.md"
        with open(summary_path, "w") as f:
            f.write(leaderboard.summary())

    # Final report
    total_time = time.time() - start_time
    log.info("")
    log.info("=" * 60)
    log.info("AutoResearch Complete!")
    log.info(f"  Total experiments: {args.max_experiments}")
    log.info(f"  Improvements found: {n_improvements}")
    log.info(f"  Failures: {n_failures}")
    log.info(f"  Champion dice: {champion_dice:.4f}")
    log.info(f"  Total time: {total_time/3600:.1f}h")
    log.info("=" * 60)

    # Print leaderboard
    print(leaderboard.summary())


if __name__ == "__main__":
    main()
