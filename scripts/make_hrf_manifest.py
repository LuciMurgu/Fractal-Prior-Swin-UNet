"""Generate HRF manifest JSONL with stem-based matching."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _scan_by_stem(root: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for path in root.iterdir():
        if path.suffix.lower() in SUPPORTED_EXTS and path.is_file():
            mapping[path.stem] = path
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Make HRF manifest JSONL.")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--images_subdir", type=str, default="images")
    parser.add_argument("--manual_subdir", type=str, default="manual1")
    parser.add_argument("--fov_subdir", type=str, default="mask")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--write_absolute_paths", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    images_dir = root / args.images_subdir
    manual_dir = root / args.manual_subdir
    fov_dir = root / args.fov_subdir

    images = _scan_by_stem(images_dir)
    manuals = _scan_by_stem(manual_dir)
    fovs = _scan_by_stem(fov_dir)

    missing = []
    records: List[dict] = []

    for stem, image_path in images.items():
        manual_path = manuals.get(stem)
        fov_path = fovs.get(stem) or fovs.get(f"{stem}_mask")
        if manual_path is None or fov_path is None:
            missing.append(stem)
            continue

        def _rel_or_abs(path: Path) -> str:
            return str(path if args.write_absolute_paths else path.relative_to(root))

        records.append(
            {
                "id": stem,
                "image_path": _rel_or_abs(image_path),
                "mask_path": _rel_or_abs(manual_path),
                "fov_mask_path": _rel_or_abs(fov_path),
                "meta": {"dataset": "HRF"},
            }
        )

    if missing:
        raise ValueError(f"Missing mask or FOV for stems: {missing}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec) + "\n")

    print(f"images: {len(images)} matched: {len(records)} missing: {len(missing)}")


if __name__ == "__main__":
    main()
