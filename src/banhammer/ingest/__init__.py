"""Offline download and normalize. `python -m banhammer.ingest`."""

from banhammer.ingest.comp_rules import CompRules
from banhammer.ingest.scryfall_cards import ScryfallCards

SOURCES = {
    "scryfall_cards": ScryfallCards(),
    "comp_rules": CompRules(),
}

__all__ = ["SOURCES", "CompRules", "ScryfallCards"]
