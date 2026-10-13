"""Stage 0: card image to text."""
from __future__ import annotations

from pathlib import Path

from banhammer.models import OCRCard


def run_ocr(image: Path | str, photo: bool | None = None, debug_dir: Path | str | None = None) -> OCRCard:
    from .pipeline import OCRPipeline
    return OCRPipeline().run(image, photo=photo, debug_dir=debug_dir)


__all__ = ["run_ocr"]
