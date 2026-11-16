"""Hide a gold card's identity so the judge cannot answer from memory.

The gold set holds famous cards: with their names in the prompt a model can recall that a card
is banned instead of reasoning from its text, which tells nothing about how the app does with a
new design. The oracle id is kept, so the pipeline can still drop the card from its own results.
"""
from __future__ import annotations

from banhammer.models import CardInput

ANONYMOUS_NAME = "Test Card"


def self_names(name: str) -> list[str]:
    """Ways a card's text names itself, longest first: "Kiki-Jiki, Mirror Breaker" and "Kiki-Jiki"."""
    names = [name] if name else []
    if ", " in name:
        names.append(name.split(", ", 1)[0])
    return names


def _replace_names(text: str, names: list[str]) -> str:
    for name in names:
        text = text.replace(name, ANONYMOUS_NAME)
    return text


def anonymize(card: CardInput) -> CardInput:
    """The same card named "Test Card", in its name, its faces and its own rules text."""
    names = self_names(card.name) + [n for face in card.card_faces for n in self_names(face.name)]
    names = sorted(set(names), key=len, reverse=True)
    faces = [
        face.model_copy(update={"name": f"{ANONYMOUS_NAME} ({index})" if face.name else "",
                                "oracle_text": _replace_names(face.oracle_text, names)})
        for index, face in enumerate(card.card_faces, start=1)
    ]
    return card.model_copy(update={
        "name": ANONYMOUS_NAME,
        "oracle_text": _replace_names(card.oracle_text, names),
        "card_faces": faces,
    })
