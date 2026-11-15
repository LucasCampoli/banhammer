"""Metrics to compare OCR output with Scryfall Oracle text.

Oracle text is the current wording, not what is printed on the card. The
normalizers below remove the templating changes that are most common in the
OCR population, so the error measures reading mistakes rather than errata:
printed card names where Oracle says "this creature", "enters the battlefield"
shortened to "enters", and the "Tribal" type renamed to "Kindred".
"""
from __future__ import annotations

import re

SELF_REFERENCE = "~"
SELF_REFERENCE_PHRASE = re.compile(
    r"\bthis (?:creature|artifact|enchantment|land|spell|card|permanent|equipment|aura|vehicle|token)\b",
    re.IGNORECASE,
)
ENTERS = re.compile(r"\b(?:enters the battlefield|comes into play)\b")
TYPE_RENAMES = {"Tribal": "Kindred"}
QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})


def levenshtein(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_a != char_b)))
        previous = current
    return previous[-1]


def cer(prediction: str, reference: str) -> float:
    """Character error rate, capped at 1 so one bad card cannot dominate a mean."""
    if not reference:
        return float(bool(prediction))
    return min(levenshtein(prediction, reference) / len(reference), 1.0)


def flatten(text: str | None) -> str:
    """Collapse whitespace and line breaks and unify typographic quotes."""
    return re.sub(r"\s+", " ", (text or "").translate(QUOTES)).strip()


def self_names(name: str) -> list[str]:
    """Ways a card refers to itself in printed text, longest first.

    Legendary cards often use the part of the name before the comma
    ("Hogaak" for "Hogaak, Arisen Necropolis").
    """
    names = {name}
    if "," in name:
        names.add(name.split(",", 1)[0])
    return sorted(names, key=len, reverse=True)


def normalize_rules(text: str | None, name: str) -> str:
    text = flatten(text)
    for self_name in self_names(name):
        text = re.sub(rf"(?<!\w){re.escape(self_name)}(?!\w)", SELF_REFERENCE, text)
    text = SELF_REFERENCE_PHRASE.sub(SELF_REFERENCE, text)
    return ENTERS.sub("enters", text)


def normalize_type_line(text: str | None) -> str:
    text = flatten(text)
    for old, new in TYPE_RENAMES.items():
        text = re.sub(rf"\b{old}\b", new, text)
    return text
