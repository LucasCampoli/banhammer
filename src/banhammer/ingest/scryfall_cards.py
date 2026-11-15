"""Scryfall oracle cards: Modern-legal and Modern-banned cards, one document each.

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

from banhammer.models import OCRCard
from banhammer.ocr.scope import in_ocr_scope
from banhammer.ocr.transcriptions import DEFAULT_TRANSCRIPTIONS, read_transcriptions
from banhammer.shared.cardtext import card_types, format_card
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

    def __init__(self, transcriptions: Path = DEFAULT_TRANSCRIPTIONS):
        self.transcriptions = transcriptions

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
        if not self.transcriptions.exists():
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


def is_ingested(card: dict) -> bool:
    """Whether a bulk card belongs to the indexed population (Modern legal or banned)."""
    return (
        card.get("layout") not in DROPPED_LAYOUTS
        and (card.get("legalities") or {}).get("modern") in {"legal", "banned"}
        and bool(card.get("oracle_id"))
    )


def card_to_document(card: dict, transcription: OCRCard | None = None) -> Document | None:
    if not is_ingested(card):
        return None
    oracle_id = card["oracle_id"]
    modern = card["legalities"]["modern"]
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
        "keywords": card.get("keywords") or [],
        "modern": modern,
        "source": SOURCE_SCRYFALL if transcription is None else SOURCE_OCR,
    })
    return Document(id=oracle_id, page_content=text, metadata=metadata)
