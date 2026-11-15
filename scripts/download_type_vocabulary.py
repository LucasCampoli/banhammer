"""Download the words that can appear in a type line from Scryfall's catalogs into data/type_vocabulary.json.

Planeswalker and battle subtypes are left out: those cards are not transcribed
by the OCR, and their names only add wrong corrections.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

from banhammer.ingest.http import headers
from banhammer.shared.config import ROOT, load_env

CATALOG_URL = "https://api.scryfall.com/catalog/{name}"
HEAD_CATALOGS = ("supertypes", "card-types")
SUBTYPE_CATALOGS = ("artifact-types", "creature-types", "enchantment-types", "land-types", "spell-types")
DELAY = 0.1  # Scryfall asks for 50-100 ms between requests


def catalog_words(session: requests.Session, name: str) -> set[str]:
    time.sleep(DELAY)
    response = session.get(CATALOG_URL.format(name=name), headers=headers(), timeout=30)
    response.raise_for_status()
    # multi-word subtypes ("Time Lord") are matched word by word
    return {word for value in response.json()["data"] for word in value.split()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "type_vocabulary.json")
    args = ap.parse_args()
    load_env()
    with requests.Session() as session:
        head = set().union(*(catalog_words(session, name) for name in HEAD_CATALOGS))
        subtypes = set().union(*(catalog_words(session, name) for name in SUBTYPE_CATALOGS))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"head": sorted(head), "subtypes": sorted(subtypes)}, indent=1))
    print(f"Saved {len(head)} supertypes and card types and {len(subtypes)} subtypes to {args.out}")


if __name__ == "__main__":
    main()
