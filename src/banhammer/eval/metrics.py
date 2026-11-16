"""Retrieval and generation scores for one pipeline configuration."""
from __future__ import annotations

from banhammer.models import BalanceReport, Hit


def card_names(hits: list[Hit]) -> list[str]:
    """Names of the card hits, in retrieval order."""
    return [str(hit.metadata["name"]) for hit in hits
            if hit.namespace == "cards" and hit.metadata.get("name")]


def recall_at_k(hits: list[Hit], pieces: list[str | list[str]], k: int = 10) -> float:
    """Fraction of combo pieces found in the top-k card hits.

    A piece is a card name, or a list of cards that fill the same role in the combo (any of
    them counts, e.g. every creature that untaps a permanent when it enters).
    """
    if not pieces:
        raise ValueError("recall is undefined when there are no combo pieces")
    names = {name.casefold() for name in card_names(hits)[:k]}
    roles = [[piece] if isinstance(piece, str) else piece for piece in pieces]
    found = sum(any(name.casefold() in names for name in role) for role in roles)
    return found / len(roles)


def risk_matches(report: BalanceReport, expected: str) -> bool:
    return report.risk == expected


def false_alarm(report: BalanceReport, expected: str) -> bool | None:
    """A card that should be `ok` but was flagged. None when the card is not expected to be ok."""
    if expected != "ok":
        return None
    return report.risk != "ok"


def ban_detected(report: BalanceReport, expected: str) -> bool | None:
    """A card that should be `likely_ban` and was. None when the card is not expected to be banned."""
    if expected != "likely_ban":
        return None
    return report.risk == "likely_ban"


def severe_false_alarm(report: BalanceReport, expected: str) -> bool | None:
    """A card that should be `ok` but was called `likely_ban`. None when the card is not expected to be ok."""
    if expected != "ok":
        return None
    return report.risk == "likely_ban"


WILSON_Z = 1.96  # 95% confidence


def wilson_interval(successes: int, n: int, z: float = WILSON_Z) -> tuple[float, float] | None:
    """95% Wilson score interval of a proportion. None when n is 0.

    With 37 gold cards a rate of 0.5 spans about 0.34 to 0.66: differences inside that range
    are not evidence that one config is better.
    """
    if n == 0:
        return None
    rate = successes / n
    denominator = 1 + z * z / n
    centre = (rate + z * z / (2 * n)) / denominator
    half = z * ((rate * (1 - rate) / n + z * z / (4 * n * n)) ** 0.5) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def mean(values: list[float | bool | None]) -> float | None:
    present = [float(v) for v in values if v is not None]
    return sum(present) / len(present) if present else None
