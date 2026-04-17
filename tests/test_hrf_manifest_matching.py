import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


def _write_image(path: Path) -> None:
    img = Image.new("RGB", (4, 4), color=(10, 20, 30))
    img.save(path)


def _write_mask(path: Path) -> None:
    img = Image.new("L", (4, 4), color=255)
    img.save(path)


def test_hrf_manifest_matching(tmp_path: Path) -> None:
    root = tmp_path / "HRF"
    images = root / "images"
    manual = root / "manual1"
    mask = root / "mask"
    images.mkdir(parents=True)
    manual.mkdir(parents=True)
    mask.mkdir(parents=True)

    _write_image(images / "img1.png")
    _write_mask(manual / "img1.tif")
    _write_mask(mask / "img1.jpg")

    out = tmp_path / "hrf.jsonl"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "scripts" / "make_hrf_manifest.py"),
        "--root",
        str(root),
        "--out",
        str(out),
    ]
    subprocess.check_call(cmd)

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["id"] == "img1"
