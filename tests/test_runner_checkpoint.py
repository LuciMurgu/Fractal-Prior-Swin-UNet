"""Tests for experiment runner checkpoint resolution."""

from __future__ import annotations

from pathlib import Path

from fractal_swin_unet.exp.runner import resolve_trained_checkpoint


def test_resolve_trained_checkpoint_prefers_best(tmp_path: Path) -> None:
    (tmp_path / "checkpoint.pt").write_text("x", encoding="utf-8")
    (tmp_path / "checkpoint_best.pt").write_text("y", encoding="utf-8")
    assert resolve_trained_checkpoint(tmp_path) == str(tmp_path / "checkpoint_best.pt")


def test_resolve_trained_checkpoint_final_only(tmp_path: Path) -> None:
    (tmp_path / "checkpoint.pt").write_text("x", encoding="utf-8")
    assert resolve_trained_checkpoint(tmp_path) == str(tmp_path / "checkpoint.pt")


def test_resolve_trained_checkpoint_missing(tmp_path: Path) -> None:
    assert resolve_trained_checkpoint(tmp_path) is None
