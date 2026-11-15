"""Scryfall oracle cards: cards legal or banned in Modern or Standard, one document each.

A card in the OCR scope is indexed with its OCR transcription when there is a
usable one for the same printing; every other card keeps its Scryfall text.
The `source` metadata says which one a document holds. Structured metadata
(name, cost, colors, legality) always comes from Scryfall: the pipeline uses
it to filter hits, so it must not depend on a reading.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import requests
from langchain_core.documents import Document

from banhammer.models import FORMATS, OCRCard
from banhammer.ocr.scope import in_ocr_scope
from banhammer.ocr.transcriptions import DEFAULT_TRANSCRIPTIONS, read_transcriptions
from banhammer.shared.cardtext import card_types, format_card, type_words
from banhammer.shared.documents import clean_metadata
from banhammer.ingest.http import download, headers

BULK_DATA_URL = "https://api.scryfall.com/bulk-data"
DROPPED_LAYOUTS = frozenset({
    "art_series",
    "token",
    "double_faced_token",
    "emblem",
    "planar",
    "scheme",
    "vanguard",
    "front_card",
})
SOURCE_OCR = "ocr"
SOURCE_SCRYFALL = "scryfall"


def open_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


class ScryfallCards:
    namespace = "cards"
    filename = "oracle-cards.jsonl.gz"

    def __init__(self, transcriptions: Path | None = DEFAULT_TRANSCRIPTIONS, require_transcriptions: bool = False):
        """`transcriptions` None indexes every card with its Scryfall text.

        With `require_transcriptions`, a missing transcriptions file is an error
        instead of a silent fallback to Scryfall text.
        """
        self.transcriptions = transcriptions
        self.require_transcriptions = require_transcriptions

    def download(self, dest: Path) -> None:
        response = requests.get(BULK_DATA_URL, headers=headers(), timeout=60)
        response.raise_for_status()
        payload = response.json()
        items = payload["data"] if isinstance(payload, dict) else payload
        match = next(item for item in items if item.get("type") == "oracle_cards")
        uri = match.get("jsonl_download_uri") or match["download_uri"]
        download(uri, dest)

    def normalize(self, raw: Path) -> list[Document]:
        transcriptions = self._load_transcriptions()
        docs: list[Document] = []
        with open_text(raw) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                card = json.loads(line)
                doc = card_to_document(card, transcription_for(card, transcriptions))
                if doc is not None:
                    docs.append(doc)
        from_ocr = sum(doc.metadata["source"] == SOURCE_OCR for doc in docs)
        print(f"scryfall_cards: {from_ocr} of {len(docs)} cards use their OCR transcription")
        return docs

    def _load_transcriptions(self) -> dict[str, tuple[str, OCRCard]]:
        if self.transcriptions is None:
            print("scryfall_cards: OCR disabled; using Scryfall text for every card")
            return {}
        if not self.transcriptions.exists():
            if self.require_transcriptions:
                raise FileNotFoundError(f"{self.transcriptions} not found; run scripts/transcribe_cards.py")
            print(f"scryfall_cards: no OCR transcriptions at {self.transcriptions}; "
                  "using Scryfall text (run scripts/transcribe_cards.py)")
            return {}
        return read_transcriptions(self.transcriptions)


def transcription_for(card: dict, transcriptions: dict[str, tuple[str, OCRCard]]) -> OCRCard | None:
    """The OCR transcription to index for a card, if the card is in scope and it was read from this printing.

    A newer bulk file may pick another printing for the same oracle id; its
    transcription would describe a different image, so it is not used.
    """
    if not in_ocr_scope(card):
        return None
    card_id, transcription = transcriptions.get(card.get("oracle_id"), (None, None))
    return transcription if card_id == card.get("id") else None


def card_formats(card: dict) -> list[str]:
    """The supported formats where the card is legal or banned."""
    legalities = card.get("legalities") or {}
    return [fmt for fmt in FORMATS if legalities.get(fmt) in {"legal", "banned"}]


def is_ingested(card: dict) -> bool:
    """Whether a bulk card belongs to the indexed population (legal or banned in a supported format)."""
    return card.get("layout") not in DROPPED_LAYOUTS and bool(card_formats(card)) and bool(card.get("oracle_id"))


def card_to_document(card: dict, transcription: OCRCard | None = None) -> Document | None:
    if not is_ingested(card):
        return None
    oracle_id = card["oracle_id"]
    legalities = card["legalities"]
    formats = card_formats(card)
    faces = card.get("card_faces") or None
    name = card.get("name") or ""
    type_line = card.get("type_line") or ""
    mana_cost = card.get("mana_cost") or ""
    oracle_text = card.get("oracle_text") or ""
    if faces and not mana_cost:
        mana_cost = " // ".join(face.get("mana_cost") or "" for face in faces)
    if transcription is not None:
        text = format_card(transcription.name, transcription.type_line, transcription.mana_cost,
                           transcription.rules_text)
    else:
        text = format_card(name, type_line, card.get("mana_cost") or "", oracle_text, faces)
    metadata = clean_metadata({
        "oracle_id": oracle_id,
        "name": name,
        "mana_cost": mana_cost,
        "cmc": card.get("cmc"),
        "colors": card.get("colors") or [],
        "types": card_types(type_line),
        "type_words": type_words(type_line),
        "keywords": card.get("keywords") or [],
        "formats": formats,
        "banned": [fmt for fmt in formats if legalities.get(fmt) == "banned"],
        "source": SOURCE_SCRYFALL if transcription is None else SOURCE_OCR,
        "text": text,
    })
    return Document(id=oracle_id, page_content=text, metadata=metadata)
