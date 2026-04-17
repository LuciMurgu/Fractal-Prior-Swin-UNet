"""Leakage audit utilities."""

from __future__ import annotations

from typing import Dict, List, Any


def assert_no_patient_leakage(samples: List[Dict[str, Any]]) -> None:
    """Ensure no patient_id appears in multiple splits.

    Args:
        samples: List of manifest samples with split and patient_id.

    Raises:
        ValueError: If leakage is detected.
    """

    patient_splits: Dict[str, set[str]] = {}
    for sample in samples:
        patient_id = sample.get("patient_id")
        split = sample.get("split")
        if patient_id is None or split is None:
            continue
        patient_splits.setdefault(str(patient_id), set()).add(str(split))

    leaking = {pid: splits for pid, splits in patient_splits.items() if len(splits) > 1}
    if leaking:
        raise ValueError(f"Patient leakage detected: {leaking}")
