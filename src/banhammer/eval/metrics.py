"""Retrieval and generation scores for one pipeline configuration."""
from __future__ import annotations

from banhammer.models import BalanceReport, Hit


def recall_at_k(hits: list[Hit], pieces: list[str], k: int = 10) -> float:
    """Fraction of known combo pieces found in the top-k card hits."""
    if not pieces:
        raise ValueError("recall is undefined when there are no combo pieces")
    names: list[str] = []
    for hit in hits:
        if hit.namespace != "cards":
            continue
        name = hit.metadata.get("name")
        if name:
            names.append(str(name).casefold())
        if len(names) == k:
            break
    found = {name.casefold() for name in pieces} & set(names)
    return len(found) / len(pieces)


def risk_matches(report: BalanceReport, expected: str) -> bool:
    return report.risk == expected
