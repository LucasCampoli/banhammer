"""Which Scryfall cards the OCR transcribes; the rest are loaded from Scryfall text.

The zones assume a single-faced card with the 2003 or 2015 frame, a text box
and an optional P/T box. Other layouts, planeswalkers and borderless or
extended-art printings place their text elsewhere.
"""
from __future__ import annotations

OCR_LAYOUTS = frozenset({"normal"})
OCR_FRAMES = frozenset({"2003", "2015"})
EXCLUDED_FRAME_EFFECTS = frozenset({"extendedart"})
EXCLUDED_BORDERS = frozenset({"borderless"})
EXCLUDED_TYPES = ("Planeswalker",)


def in_ocr_scope(card: dict) -> bool:
    """Whether the OCR supports this Scryfall card object."""
    if card.get("layout") not in OCR_LAYOUTS or card.get("frame") not in OCR_FRAMES:
        return False
    if card.get("border_color") in EXCLUDED_BORDERS:
        return False
    if EXCLUDED_FRAME_EFFECTS & set(card.get("frame_effects") or ()):
        return False
    type_line = card.get("type_line") or ""
    return not any(t in type_line for t in EXCLUDED_TYPES)
