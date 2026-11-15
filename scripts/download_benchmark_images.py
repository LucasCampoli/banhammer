"""Download the Scryfall `large` images of the OCR benchmark cards into images/benchmark/ (gitignored)."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from banhammer.ingest.http import download
from banhammer.ocr.benchmark import DEFAULT_BENCHMARK, DEFAULT_BULK, DEFAULT_IMAGES, cards_by_id, read_benchmark
from banhammer.shared.config import load_env

DELAY = 0.1  # Scryfall asks for 50-100 ms between requests


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bulk", type=Path, default=DEFAULT_BULK)
    ap.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    ap.add_argument("--out", type=Path, default=DEFAULT_IMAGES)
    args = ap.parse_args()
    load_env()
    entries = read_benchmark(args.benchmark)
    cards = cards_by_id(args.bulk, {e["id"] for e in entries})
    downloaded = 0
    for entry in entries:
        dest = args.out / f"{entry['id']}.jpg"
        if dest.exists():
            continue
        time.sleep(DELAY)
        download(cards[entry["id"]]["image_uris"]["large"], dest)
        downloaded += 1
    print(f"Downloaded {downloaded} images, {len(entries) - downloaded} already present in {args.out}")


if __name__ == "__main__":
    main()
