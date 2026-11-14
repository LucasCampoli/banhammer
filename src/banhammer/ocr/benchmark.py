"""OCR benchmark: a fixed sample of the cards the OCR supports.

Only card ids and their strata are versioned (`data/eval/ocr_benchmark.jsonl`).
The expected text comes from the Scryfall bulk file and the images from
Scryfall, so no card text or art is committed.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path

from banhammer.ingest.scryfall_cards import is_ingested, open_text
from banhammer.shared.config import ROOT
from .scope import in_ocr_scope

DEFAULT_BULK = ROOT / "data" / "raw" / "oracle-cards.jsonl.gz"
DEFAULT_BENCHMARK = ROOT / "data" / "eval" / "ocr_benchmark.jsonl"
DEFAULT_IMAGES = ROOT / "images" / "benchmark"
UNIFORM = "uniform"
LONG_RULES_CHARS = 300
LARGE_COST_SYMBOLS = 5


def _effects(card: dict) -> set[str]:
    return set(card.get("frame_effects") or ())


def _effect(name: str) -> Callable[[dict], bool]:
    return lambda card: name in _effects(card)


# Card features that may break the OCR and are too rare for a uniform sample to cover.
STRATA: dict[str, Callable[[dict], bool]] = {
    "frame_2003": lambda c: c["frame"] == "2003",
    "legendary": _effect("legendary"),
    "nyx": _effect("enchantment"),
    "devoid": _effect("devoid"),
    "snow": _effect("snow"),
    "lesson": _effect("lesson"),
    "spree": _effect("spree"),
    "colorshifted": _effect("colorshifted"),
    "miracle": _effect("miracle"),
    "companion": _effect("companion"),
    "white_border": lambda c: c.get("border_color") == "white",
    "lowres": lambda c: c.get("image_status") == "lowres",
    "vehicle": lambda c: "Vehicle" in c["type_line"],
    "land": lambda c: "Land" in c["type_line"],
    "no_rules_text": lambda c: not c.get("oracle_text"),
    "long_rules_text": lambda c: len(c.get("oracle_text") or "") > LONG_RULES_CHARS,
    "large_cost": lambda c: (c.get("mana_cost") or "").count("{") >= LARGE_COST_SYMBOLS,
}


def iter_cards(bulk: Path) -> Iterator[dict]:
    with open_text(bulk) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_population(bulk: Path) -> list[dict]:
    """Cards that are both ingested and supported by the OCR."""
    return [card for card in iter_cards(bulk) if is_ingested(card) and in_ocr_scope(card)]


def read_benchmark(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_benchmark(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")


def cards_by_id(bulk: Path, ids: set[str]) -> dict[str, dict]:
    cards = {card["id"]: card for card in iter_cards(bulk) if card["id"] in ids}
    missing = ids - cards.keys()
    if missing:
        raise ValueError(f"{len(missing)} benchmark cards are not in {bulk}, e.g. {sorted(missing)[:3]}")
    return cards
