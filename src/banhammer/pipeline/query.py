"""Query building: the version 1 queries, plus LLM decomposition and query expansion."""
from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from banhammer.models import Ability, CardAnalysis, CardInput, Query
from banhammer.pipeline.retrieval import format_filter

MAX_COUNTERPARTS = 5
MAX_RULES_TOPICS = 5

_DECOMPOSE_SYSTEM = """You are a Magic: The Gathering rules and combo expert.
You are given a newly designed card. Do not judge it; break it down so we can search for what it
interacts with.

- abilities: split every ability into trigger, cost, effect and restrictions. Leave a field empty
  when the ability has none.
- counterparts: up to 5 mechanical counterparts. Each one describes, as English Oracle rules text,
  the OTHER half of a combo: an existing card that would turn this card into an infinite loop, a
  lock or a degenerate engine. Do not describe cards that do the same thing as this card.
  Example: if the card triggers whenever an opponent draws a card, a counterpart is
  "At the beginning of each player's upkeep, that player draws a card." or
  "Each opponent draws a card."
  Give the counterpart's card type and a maximum mana value when the combo needs them.
- rules_topics: up to 5 Comprehensive Rules topics the card's wording depends on, such as
  keywords, token types, timing restrictions or replacement effects."""


def base_queries(card: CardInput, fmt: str, use_format_filter: bool = True) -> list[Query]:
    """Version 1: the whole card text, once against the catalog and once against the rules."""
    text = card.as_text()
    return [
        Query(text=text, namespace="cards", filter=format_filter(fmt) if use_format_filter else None),
        Query(text=text, namespace="rules"),
    ]


def decompose(llm: Any, card: CardInput) -> CardAnalysis:
    """Ask the LLM to split the card into primitives, counterparts and rules topics."""
    result = llm.with_structured_output(CardAnalysis).invoke([
        SystemMessage(content=_DECOMPOSE_SYSTEM),
        HumanMessage(content=f"Card:\n{card.as_text()}"),
    ])
    return result if isinstance(result, CardAnalysis) else CardAnalysis.model_validate(result)


def ability_text(ability: Ability) -> str:
    """The ability followed by its non-empty primitives, as one search string."""
    parts = [ability.text]
    for label, value in (("trigger", ability.trigger), ("cost", ability.cost),
                         ("effect", ability.effect), ("restrictions", ability.restrictions)):
        if value:
            parts.append(f"{label}: {value}")
    return " | ".join(parts)


def expanded_queries(
    card: CardInput,
    analysis: CardAnalysis,
    fmt: str,
    use_format_filter: bool = True,
) -> list[Query]:
    """Version 1 queries plus one rules query per ability and topic, one catalog query per counterpart."""
    queries = base_queries(card, fmt, use_format_filter)
    queries += [Query(text=ability_text(ability), namespace="rules") for ability in analysis.abilities]
    queries += [Query(text=topic, namespace="rules") for topic in analysis.rules_topics[:MAX_RULES_TOPICS]]
    card_filter = format_filter(fmt) if use_format_filter else None
    queries += [
        Query(text=counterpart.description, namespace="cards", filter=card_filter)
        for counterpart in analysis.counterparts[:MAX_COUNTERPARTS]
    ]
    return queries


def build_queries(card: CardInput, fmt: str, cfg: dict, llm: Any) -> tuple[list[Query], CardAnalysis | None]:
    """The queries for one card and, with ``decompose: true``, the analysis they came from."""
    use_format_filter = bool(cfg.get("format_filter", True))
    if not cfg.get("decompose", False):
        return base_queries(card, fmt, use_format_filter), None
    analysis = decompose(llm, card)
    return expanded_queries(card, analysis, fmt, use_format_filter), analysis
