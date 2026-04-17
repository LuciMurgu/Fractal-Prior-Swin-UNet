import json
from pathlib import Path

from fractal_swin_unet.exp.report import generate_summary


def test_report_aggregation(tmp_path: Path) -> None:
    run_a = tmp_path / "runA"
    run_b = tmp_path / "runB"
    run_a.mkdir()
    run_b.mkdir()

    (run_a / "metrics.json").write_text(json.dumps({"train_dice": 0.7}), encoding="utf-8")
    (run_b / "metrics.json").write_text(json.dumps({"train_dice": 0.5}), encoding="utf-8")

    (run_a / "threshold.json").write_text(json.dumps({"best_dice": 0.8, "dice_at_0_5": 0.7, "tau_star": 0.5}), encoding="utf-8")
    (run_b / "threshold.json").write_text(json.dumps({"best_dice": 0.6, "dice_at_0_5": 0.5, "tau_star": 0.4}), encoding="utf-8")

    reports = generate_summary({"expA": run_a, "expB": run_b}, tmp_path, "matrix")

    assert reports["csv"].exists()
    assert reports["md"].exists()
    assert reports["json"].exists()

    md_text = reports["md"].read_text(encoding="utf-8")
    assert "expA" in md_text
    assert "expB" in md_text
