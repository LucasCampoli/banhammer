"""One query per namespace, using the whole card text."""
from __future__ import annotations

from banhammer.models import CardInput, Query
from banhammer.shared.registry import register


@register("build_query", "full_text")
class FullText:
    def build(self, card: CardInput) -> list[Query]:
        text = card.as_text()
        return [Query(text=text, namespace="cards"), Query(text=text, namespace="rules")]
