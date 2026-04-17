import pytest

from fractal_swin_unet.exp.audit import assert_no_patient_leakage
from fractal_swin_unet.exp.split import assign_splits


def test_patient_split_no_leakage() -> None:
    samples = [
        {"id": "a1", "image_path": "x", "patient_id": "p1"},
        {"id": "a2", "image_path": "x", "patient_id": "p1"},
        {"id": "b1", "image_path": "x", "patient_id": "p2"},
        {"id": "c1", "image_path": "x", "patient_id": "p3"},
    ]

    assigned = assign_splits(
        samples,
        mode="patient",
        seed=123,
        ratios={"train": 0.5, "val": 0.25, "test": 0.25},
    )

    patient_splits = {}
    for sample in assigned:
        patient_splits.setdefault(sample["patient_id"], set()).add(sample["split"])

    assert all(len(splits) == 1 for splits in patient_splits.values())


def test_patient_leakage_audit_raises() -> None:
    samples = [
        {"id": "a1", "image_path": "x", "patient_id": "p1", "split": "train"},
        {"id": "a2", "image_path": "x", "patient_id": "p1", "split": "val"},
    ]

    with pytest.raises(ValueError):
        assert_no_patient_leakage(samples)
