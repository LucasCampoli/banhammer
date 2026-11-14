"""Download the Tesseract `tessdata_best` model the OCR uses into data/tessdata/ (gitignored).

The best (float) LSTM models read the benchmark more accurately than the fast
models that system packages ship, at about 20% more time per card.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from banhammer.ingest.http import download
from banhammer.shared.config import ROOT, load_env

RELEASE = "4.1.0"
URL = "https://github.com/tesseract-ocr/tessdata_best/raw/{release}/{lang}.traineddata"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "tessdata")
    ap.add_argument("--lang", default="eng")
    args = ap.parse_args()
    load_env()
    dest = args.out / f"{args.lang}.traineddata"
    download(URL.format(release=RELEASE, lang=args.lang), dest)
    print(f"Saved tessdata_best {RELEASE} {args.lang} model to {dest}")


if __name__ == "__main__":
    main()
