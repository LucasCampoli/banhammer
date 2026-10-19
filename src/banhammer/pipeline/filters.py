"""Drop hits that should not reach the judge."""
from __future__ import annotations

from banhammer.models import CardInput, Hit
from banhammer.shared.registry import register


@register("filter", "exclude_self")
class ExcludeSelf:
    """A real card being re-checked should not come back as its own result."""

    def apply(self, card: CardInput, hits: list[Hit]) -> list[Hit]:
        names = card.names()
        kept = []
        for hit in hits:
            name = str(hit.metadata.get("name", "")).casefold()
            if name and name in names:
                continue
            kept.append(hit)
        return kept
