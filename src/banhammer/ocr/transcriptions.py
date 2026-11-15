"""OCR transcriptions of the card catalog, read by the ingest.

`scripts/transcribe_cards.py` runs the OCR on the Scryfall image of every card
in scope and writes one JSON line per card: the printing's id, its oracle id
and either the OCR result or the error that stopped it. The ingest indexes a
transcription only when it is usable; any other card keeps its Scryfall text,
so a failed or missing transcription never drops a card from the index.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TextIO

from banhammer.models import OCRCard
from banhammer.shared.config import ROOT

DEFAULT_TRANSCRIPTIONS = ROOT / "data" / "processed" / "ocr_cards.jsonl"


def write_transcription(handle: TextIO, card_id: str, oracle_id: str, result: OCRCard | Exception) -> None:
    row: dict = {"id": card_id, "oracle_id": oracle_id}
    if isinstance(result, Exception):
        row["error"] = f"{type(result).__name__}: {result}"
    else:
        row["card"] = result.model_dump()
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()  # a long run can be stopped and resumed without losing finished cards


def transcribed_ids(path: Path) -> set[str]:
    """Card ids with a line in `path`, failed ones included, so a resumed run skips them."""
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as handle:
        return {json.loads(line)["id"] for line in handle if line.strip()}


def is_usable(card: OCRCard) -> bool:
    """Whether a transcription is complete enough to replace the Scryfall text.

    Every card in scope prints a name and a type line, so an empty one means the
    OCR lost a zone. A card with no rules text is possible (vanilla creatures).
    """
    return bool(card.name.strip() and card.type_line.strip())


def read_transcriptions(path: Path) -> dict[str, tuple[str, OCRCard]]:
    """Usable transcriptions by oracle id, with the id of the printing they were read from."""
    out: dict[str, tuple[str, OCRCard]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "card" not in row:
                continue
            card = OCRCard.model_validate(row["card"])
            if is_usable(card):
                out[row["oracle_id"]] = (row["id"], card)
    return out
