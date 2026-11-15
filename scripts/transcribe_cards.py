"""Transcribe every card in the OCR scope into data/processed/ocr_cards.jsonl for the ingest.

Downloads each card's Scryfall `large` image into images/cards/ (gitignored),
then runs the OCR on all of them in parallel. The output is appended to and
flushed per card, so a stopped run resumes where it left off. A card whose
image or OCR fails gets an error line and keeps its Scryfall text in the index.
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import requests

from banhammer.ingest.http import download
from banhammer.models import OCRCard
from banhammer.ocr.benchmark import DEFAULT_BULK, load_population
from banhammer.ocr.pipeline import DEFAULT_CONFIG, OCRPipeline, load_config
from banhammer.ocr.transcriptions import DEFAULT_TRANSCRIPTIONS, transcribed_ids, write_transcription
from banhammer.shared.config import ROOT, load_env

DEFAULT_IMAGES = ROOT / "images" / "cards"
DELAY = 0.1  # Scryfall asks for 50-100 ms between requests
PROGRESS_EVERY = 500

_pipeline: OCRPipeline | None = None


def _init_worker(config_path: Path) -> None:
    global _pipeline
    _pipeline = OCRPipeline(load_config(config_path))


def _transcribe(image: Path, frame: str) -> OCRCard:
    return _pipeline.run(image, frame=frame)


def download_images(cards: list[dict], images: Path) -> dict[str, Exception]:
    """Download the missing images; returns the download error of each card that failed."""
    errors: dict[str, Exception] = {}
    for card in cards:
        dest = images / f"{card['id']}.jpg"
        if dest.exists():
            continue
        time.sleep(DELAY)
        try:
            download(card["image_uris"]["large"], dest)
        except (requests.RequestException, KeyError) as error:
            errors[card["id"]] = error
    return errors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bulk", type=Path, default=DEFAULT_BULK)
    ap.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    ap.add_argument("--out", type=Path, default=DEFAULT_TRANSCRIPTIONS)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="OCR config")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, help="transcribe at most this many new cards (for a trial run)")
    args = ap.parse_args()
    load_env()

    done = transcribed_ids(args.out)
    cards = [card for card in load_population(args.bulk) if card["id"] not in done]
    cards = cards[:args.limit] if args.limit else cards
    print(f"{len(done)} cards already transcribed, {len(cards)} to go")

    download_errors = download_images(cards, args.images)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    os.environ["OMP_THREAD_LIMIT"] = "1"  # one Tesseract thread per worker process
    failed = len(download_errors)
    with args.out.open("a", encoding="utf-8") as handle, \
            ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(args.config,)) as pool:
        for card in cards:
            if card["id"] in download_errors:
                write_transcription(handle, card["id"], card["oracle_id"], download_errors[card["id"]])
        jobs = {pool.submit(_transcribe, args.images / f"{card['id']}.jpg", card["frame"]): card
                for card in cards if card["id"] not in download_errors}
        for finished, job in enumerate(as_completed(jobs), 1):
            card = jobs[job]
            try:
                result: OCRCard | Exception = job.result()
            except Exception as error:  # noqa: BLE001 - recorded per card; one bad image must not stop the run
                result = error
                failed += 1
            write_transcription(handle, card["id"], card["oracle_id"], result)
            if finished % PROGRESS_EVERY == 0:
                print(f"{finished}/{len(jobs)} transcribed")
    print(f"Transcribed {len(cards) - failed} cards, {failed} failed (see the error lines in {args.out})")


if __name__ == "__main__":
    main()
