"""Download and normalize the card catalog and the Comprehensive Rules."""
from __future__ import annotations

import argparse
from pathlib import Path

from banhammer.shared.config import ROOT, load_env, load_yaml
from banhammer.shared.documents import write_jsonl
from banhammer.ingest import SOURCES, ScryfallCards

CARD_TEXT_OCR = "ocr"
CARD_TEXT_SCRYFALL = "scryfall"


def card_source(card_text: str) -> ScryfallCards:
    if card_text == CARD_TEXT_SCRYFALL:
        return ScryfallCards(transcriptions=None)
    return ScryfallCards(require_transcriptions=True)


def run(config_path: Path, force: bool = False, only: str | None = None, card_text: str | None = None) -> None:
    """`card_text` picks where OCR-scope cards take their text from and rebuilds the card records.

    None keeps the default: OCR transcriptions if they exist, Scryfall text otherwise.
    """
    load_env()
    cfg = load_yaml(config_path)
    raw_dir = ROOT / cfg.get("raw_dir", "data/raw")
    processed_dir = ROOT / cfg.get("processed_dir", "data/processed")
    raw_dir.mkdir(parents=True, exist_ok=True)
    selected = [only] if only else list(cfg["sources"])
    sources = dict(SOURCES)
    if card_text:
        sources["scryfall_cards"] = card_source(card_text)
    for name in selected:
        source = sources[name]
        raw_path = raw_dir / source.filename
        out_path = processed_dir / f"{source.namespace}.jsonl"
        if raw_path.exists() and not force:
            print(f"{name}: raw file exists, skipping download ({raw_path})")
        else:
            print(f"{name}: downloading")
            source.download(raw_path)
        rebuild_cards = card_text is not None and name == "scryfall_cards"
        if out_path.exists() and not force and not rebuild_cards:
            print(f"{name}: processed file exists, skipping normalize ({out_path})")
            continue
        docs = source.normalize(raw_path)
        write_jsonl(out_path, docs)
        print(f"{name}: wrote {len(docs)} records to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m banhammer.ingest", description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "ingest.yaml")
    parser.add_argument("--force", action="store_true", help="download and normalize even if the files exist")
    parser.add_argument("--only", choices=sorted(SOURCES))
    parser.add_argument(
        "--card-text", choices=(CARD_TEXT_OCR, CARD_TEXT_SCRYFALL),
        help="text of the cards the OCR supports: their OCR transcription (falls back to Scryfall per card) "
             "or Scryfall for every card; rebuilds the card records. Default: OCR if transcribed",
    )
    args = parser.parse_args()
    run(args.config, force=args.force, only=args.only, card_text=args.card_text)


if __name__ == "__main__":
    main()
