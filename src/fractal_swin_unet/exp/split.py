"""Deterministic split assignment utilities."""

from __future__ import annotations

import random
from typing import Dict, List, Any


def assign_splits(
    samples: List[Dict[str, Any]],
    mode: str,
    seed: int,
    ratios: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Assign deterministic splits to samples.

    Args:
        samples: List of manifest samples.
        mode: "sample" or "patient".
        seed: RNG seed.
        ratios: Dict with train/val/test ratios summing to 1.0.

    Returns:
        New list of samples with split assigned.
    """

    if mode not in {"sample", "patient"}:
        raise ValueError("mode must be 'sample' or 'patient'.")

    total_ratio = sum(ratios.values())
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0.")

    rng = random.Random(seed)

    if mode == "sample":
        indices = list(range(len(samples)))
        rng.shuffle(indices)
        ordered = [samples[i] for i in indices]
        return _assign_by_order(ordered, ratios)

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for sample in samples:
        patient_id = sample.get("patient_id") or sample.get("id")
        groups.setdefault(str(patient_id), []).append(sample)

    group_keys = list(groups.keys())
    rng.shuffle(group_keys)

    return _assign_by_groups(groups, group_keys, ratios)


def _assign_by_order(samples: List[Dict[str, Any]], ratios: Dict[str, float]) -> List[Dict[str, Any]]:
    n = len(samples)
    train_count = int(round(n * ratios.get("train", 0.0)))
    val_count = int(round(n * ratios.get("val", 0.0)))

    # Ensure total counts do not exceed n
    if train_count + val_count > n:
        val_count = max(0, n - train_count)

    test_count = n - train_count - val_count

    splits = ("train", train_count), ("val", val_count), ("test", test_count)

    assigned: List[Dict[str, Any]] = []
    idx = 0
    for split_name, count in splits:
        for _ in range(count):
            entry = dict(samples[idx])
            entry["split"] = split_name
            assigned.append(entry)
            idx += 1

    return assigned


def _assign_by_groups(
    groups: Dict[str, List[Dict[str, Any]]],
    group_keys: List[str],
    ratios: Dict[str, float],
) -> List[Dict[str, Any]]:
    total = sum(len(groups[key]) for key in group_keys)
    train_target = int(round(total * ratios.get("train", 0.0)))
    val_target = int(round(total * ratios.get("val", 0.0)))
    if train_target + val_target > total:
        val_target = max(0, total - train_target)
    targets = [train_target, val_target, total - train_target - val_target]
    split_names = ["train", "val", "test"]

    assigned: List[Dict[str, Any]] = []
    split_idx = 0
    current_count = 0

    for key in group_keys:
        group = groups[key]
        group_size = len(group)
        if split_idx < len(targets) - 1 and current_count + group_size >= targets[split_idx]:
            split_idx += 1
            current_count = 0
        for sample in group:
            entry = dict(sample)
            entry["split"] = split_names[split_idx]
            assigned.append(entry)
        current_count += group_size

    return assigned
