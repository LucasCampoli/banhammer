"""Correct a type line read by OCR against the closed vocabulary of Magic types.

A type line is supertypes and card types ("Legendary Creature"), a dash, then
subtypes ("Elf Warrior"). Every word comes from a fixed list, so a misread word
close to exactly one known word is replaced with it ("EIf" -> "Elf").
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .evaluation import levenshtein

DASH = "—"
DASH_READINGS = "-—–"
SHORT_WORD = 5  # words up to this length may be one edit away from the vocabulary; longer, two


@dataclass(frozen=True)
class TypeVocabulary:
    head: frozenset[str]  # supertypes and card types, before the dash
    subtypes: frozenset[str]  # after the dash

    @property
    def words(self) -> frozenset[str]:
        return self.head | self.subtypes

    @classmethod
    def load(cls, path: Path) -> TypeVocabulary:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run scripts/download_type_vocabulary.py")
        data = json.loads(path.read_text())
        return cls(frozenset(data["head"]), frozenset(data["subtypes"]))


def correct_word(word: str, vocabulary: frozenset[str]) -> str:
    """The only vocabulary word within the allowed edit distance of `word`, or `word` itself."""
    if word in vocabulary:
        return word
    limit = 1 if len(word) <= SHORT_WORD else 2
    distances = sorted((levenshtein(word.lower(), known.lower()), known) for known in vocabulary)
    (best, known), runner_up = distances[0], distances[1][0] if len(distances) > 1 else None
    return known if best <= limit and runner_up != best else word


def _tokens(text: str) -> list[str]:
    """Words and dashes, with dashes split off the words they were read attached to."""
    tokens: list[str] = []
    for raw in text.split():
        word = raw.strip(DASH_READINGS)
        if raw[0] in DASH_READINGS:
            tokens.append(DASH)
        if word:
            tokens.append(word)
        if word and raw[-1] in DASH_READINGS:
            tokens.append(DASH)
    return [t for i, t in enumerate(tokens) if not (t == DASH and i and tokens[i - 1] == DASH)]


def correct_type_line(text: str, vocabulary: TypeVocabulary) -> str:
    tokens = [t if t == DASH else correct_word(t, vocabulary.words) for t in _tokens(text)]
    if DASH not in tokens:  # the dash was not read: put it before the first subtype
        for i, token in enumerate(tokens):
            if token not in vocabulary.head:
                if i and token in vocabulary.subtypes:
                    tokens.insert(i, DASH)
                break
    return " ".join(tokens)
