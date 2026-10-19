"""Card text and mana-cost helpers shared by ingestion and the pipeline."""
from __future__ import annotations

import re

_SYMBOL = re.compile(r"\{([^}]+)\}")
_COLOR_ORDER = ("W", "U", "B", "R", "G")
_DASH = re.compile(r"\s+[—–-]\s+")


def format_face(name: str, type_line: str, mana_cost: str, oracle_text: str) -> str:
    return f"{name} | {type_line} | {mana_cost} | {oracle_text}"


def format_card(
    name: str,
    type_line: str,
    mana_cost: str,
    oracle_text: str,
    faces: list[dict] | None = None,
) -> str:
    """One searchable string. Faces of double-faced, split and adventure cards are joined."""
    if faces:
        return " // ".join(
            format_face(
                f.get("name") or "",
                f.get("type_line") or "",
                f.get("mana_cost") or "",
                f.get("oracle_text") or "",
            )
            for f in faces
        )
    return format_face(name, type_line, mana_cost, oracle_text)


def card_types(type_line: str) -> str:
    """The part of each face's type line before the dash."""
    faces = []
    for part in type_line.split("//"):
        faces.append(_DASH.split(part.strip(), maxsplit=1)[0].strip())
    return " // ".join(faces)


def mana_value(cost: str) -> float:
    """Mana value of a cost such as ``{2}{W}{W}`` or ``{2/W}``. ``{X}`` adds nothing."""
    total = 0.0
    for raw in _SYMBOL.findall(cost or ""):
        symbol = raw.upper()
        if symbol in _COLOR_ORDER or symbol in {"C", "S"}:
            total += 1
            continue
        if "/" in symbol:
            numbers = [float(part) for part in symbol.split("/") if part.replace(".", "", 1).isdigit()]
            total += max(numbers) if numbers else 1
            continue
        try:
            total += float(symbol)
        except ValueError:
            continue
    return total


def colors_in(cost: str) -> list[str]:
    found: set[str] = set()
    for raw in _SYMBOL.findall(cost or ""):
        for part in raw.upper().split("/"):
            if part in _COLOR_ORDER:
                found.add(part)
    return [color for color in _COLOR_ORDER if color in found]
