"""Extract text from docx specs and write a markdown summary."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from docx import Document

DOC_PATHS = [
    Path("/mnt/data/Fractal-Prior Swin-UNet_ Bridging the Topology Gap in Retinal Vessel Segmentation via Geometric Inductive Biases.docx"),
    Path("/mnt/data/Fractal-Prior Swin-UNet Deep Dive (1).docx"),
]
OUTPUT_PATH = Path("docs/SPEC_SUMMARY.md")
MAX_CHARS = 12000


def _read_docx_text(path: Path) -> str:
    doc = Document(path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n[Truncated]"


def _build_summary(sections: Iterable[tuple[str, str]]) -> str:
    lines = ["# Spec Summary", "", "Generated via `python scripts/extract_spec.py`.", ""]
    for title, body in sections:
        lines.append(f"## {title}")
        lines.append("")
        lines.append(body)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    sections: list[tuple[str, str]] = []
    for path in DOC_PATHS:
        if not path.exists():
            sections.append((path.name, f"Missing docx file: {path}"))
            continue
        text = _read_docx_text(path)
        sections.append((path.name, _truncate(text, MAX_CHARS)))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(_build_summary(sections), encoding="utf-8")


if __name__ == "__main__":
    main()
