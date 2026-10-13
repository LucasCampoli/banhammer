"""CLI: python -m banhammer.ocr path/to/card.jpg [--debug] [--photo]"""
from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import OCRPipeline


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m banhammer.ocr", description="OCR a Magic card image.")
    ap.add_argument("image", type=Path)
    ap.add_argument("--debug", action="store_true", help="save zone overlay and preprocessed crops")
    ap.add_argument("--debug-dir", type=Path, default=None, help="default: outputs/debug/<image stem>")
    ap.add_argument("--photo", action="store_true", help="apply perspective correction (real photos)")
    args = ap.parse_args()
    debug_dir = (args.debug_dir or Path("outputs/debug") / args.image.stem) if (args.debug or args.debug_dir) else None
    card = OCRPipeline().run(args.image, photo=args.photo or None, debug_dir=debug_dir)
    print(card.model_dump_json(indent=2))
    if debug_dir:
        print(f"Debug files saved to {debug_dir}")


if __name__ == "__main__":
    main()
