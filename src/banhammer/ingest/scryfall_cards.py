"""Scryfall oracle cards: Modern-legal and Modern-banned cards, one document each."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import requests
from langchain_core.documents import Document

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


def open_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


class ScryfallCards:
    namespace = "cards"
    filename = "oracle-cards.jsonl.gz"

    def download(self, dest: Path) -> None:
        response = requests.get(BULK_DATA_URL, headers=headers(), timeout=60)
        response.raise_for_status()
        payload = response.json()
        items = payload["data"] if isinstance(payload, dict) else payload
        match = next(item for item in items if item.get("type") == "oracle_cards")
        uri = match.get("jsonl_download_uri") or match["download_uri"]
        download(uri, dest)

    def normalize(self, raw: Path) -> list[Document]:
        docs: list[Document] = []
        with open_text(raw) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                doc = card_to_document(json.loads(line))
                if doc is not None:
                    docs.append(doc)
        return docs


def card_to_document(card: dict) -> Document | None:
    if card.get("layout") in DROPPED_LAYOUTS:
        return None
    modern = (card.get("legalities") or {}).get("modern")
    if modern not in {"legal", "banned"}:
        return None
    oracle_id = card.get("oracle_id")
    if not oracle_id:
        return None
    faces = card.get("card_faces") or None
    name = card.get("name") or ""
    type_line = card.get("type_line") or ""
    mana_cost = card.get("mana_cost") or ""
    oracle_text = card.get("oracle_text") or ""
    if faces and not mana_cost:
        mana_cost = " // ".join(face.get("mana_cost") or "" for face in faces)
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
        "text": text,
    })
    return Document(id=oracle_id, page_content=text, metadata=metadata)
