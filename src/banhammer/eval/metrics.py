"""Retrieval and generation scores for one pipeline configuration."""
from __future__ import annotations

from banhammer.models import BalanceReport, Hit


def card_names(hits: list[Hit]) -> list[str]:
    """Names of the card hits, in retrieval order."""
    return [str(hit.metadata["name"]) for hit in hits
            if hit.namespace == "cards" and hit.metadata.get("name")]


def recall_at_k(hits: list[Hit], pieces: list[str], k: int = 10) -> float:
    """Fraction of known combo pieces found in the top-k card hits."""
    if not pieces:
        raise ValueError("recall is undefined when there are no combo pieces")
    names = {name.casefold() for name in card_names(hits)[:k]}
    found = {name.casefold() for name in pieces} & names
    return len(found) / len(pieces)


def risk_matches(report: BalanceReport, expected: str) -> bool:
    return report.risk == expected


def false_alarm(report: BalanceReport, expected: str) -> bool | None:
    """A card that should be `ok` but was flagged. None when the card is not expected to be ok."""
    if expected != "ok":
        return None
    return report.risk != "ok"


def mean(values: list[float | bool | None]) -> float | None:
    present = [float(v) for v in values if v is not None]
    return sum(present) / len(present) if present else None
