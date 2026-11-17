"""Query building: the version 1 queries, plus LLM decomposition and query expansion."""
from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from banhammer.models import Ability, CardAnalysis, CardInput, Query
from banhammer.pipeline.retrieval import card_filter

MAX_COUNTERPARTS = 5
MAX_RULES_TOPICS = 5

_DECOMPOSE_SYSTEM = """You are a Magic: The Gathering rules and combo expert.
You are given a newly designed card. Do not judge it; break it down so we can search a card
database for the OTHER half of a combo: the existing cards it combos with.

- abilities: split every ability into trigger, cost, effect and restrictions. Leave a field empty
  when the ability has none.
- counterparts: up to 5 different cards that would complete a combo with this card (an infinite
  loop, a lock, or a fast win). First ask what this card needs that it does not provide itself
  (an untap effect, a sacrifice outlet, a way to remove counters, a free recursion target, a
  payoff for a resource it makes...). Then write each counterpart as the Oracle text of a
  realistic card that provides it: a creature with an enters trigger, an instant, an artifact,
  a land with an activated ability, whatever such a card would normally be.
  Never repeat this card's own effect, never mention this card by name, and do not describe a
  rules change that no single card would print.
  card_type: the type such a card would most likely have. Most combo pieces are creatures,
  artifacts, instants or lands; use null when you are not sure.
  max_mana_value: null unless the combo clearly fails above some cost.
- rules_topics: up to 5 Comprehensive Rules topics the card's wording depends on, such as
  keywords, token types, timing restrictions or replacement effects."""


def base_queries(card: CardInput, fmt: str, use_format_filter: bool = True) -> list[Query]:
    """Version 1: the whole card text, once against the catalog and once against the rules."""
    text = card.as_text()
    return [
        Query(text=text, namespace="cards", filter=card_filter(fmt) if use_format_filter else None),
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
    use_type_mana_filter: bool = False,
) -> list[Query]:
    """Version 1 queries plus one rules query per ability and topic, one catalog query per counterpart.

    With ``use_type_mana_filter`` a counterpart query is also restricted to the card type and
    maximum mana value the analysis gave for it.
    """
    queries = base_queries(card, fmt, use_format_filter)
    queries += [Query(text=ability_text(ability), namespace="rules") for ability in analysis.abilities]
    queries += [Query(text=topic, namespace="rules") for topic in analysis.rules_topics[:MAX_RULES_TOPICS]]
    fmt_or_none = fmt if use_format_filter else None
    for counterpart in analysis.counterparts[:MAX_COUNTERPARTS]:
        if use_type_mana_filter:
            flt = card_filter(fmt_or_none, counterpart.card_type, counterpart.max_mana_value)
        else:
            flt = card_filter(fmt_or_none)
        queries.append(Query(text=counterpart.description, namespace="cards", filter=flt, purpose="combo"))
    return queries


def build_queries(card: CardInput, fmt: str, cfg: dict, llm: Any) -> tuple[list[Query], CardAnalysis | None]:
    """The queries for one card and, with ``decompose: true``, the analysis they came from."""
    use_format_filter = bool(cfg.get("format_filter", True))
    if not cfg.get("decompose", False):
        return base_queries(card, fmt, use_format_filter), None
    analysis = decompose(llm, card)
    use_type_mana_filter = bool(cfg.get("type_mana_filter", False))
    return expanded_queries(card, analysis, fmt, use_format_filter, use_type_mana_filter), analysis
