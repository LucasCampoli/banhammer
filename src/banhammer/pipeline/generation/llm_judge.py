"""Ask the LLM for a BalanceReport grounded in the retrieved hits."""
from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from banhammer.models import BalanceReport, CardInput, Hit
from banhammer.shared.registry import register

_SYSTEM = """You are a Magic: The Gathering judge for the Modern format.
You are given a card and numbered reference hits. Card hits are labeled [C1], [C2], ...
and Comprehensive Rules hits are labeled [R1], [R2], ...

Write a balance report:
- wording_issues: problems with how the card is written. Cite the rule labels you rely on in rule_ids.
- combos: dangerous interactions. Cite the card labels involved in card_ids and list the sequence of plays.
- power_assessment: power versus cost compared with the comparable cards retrieved. Cite those cards in comparable_ids.
- risk: ok, watch, or likely_ban, and a confidence between 0 and 1.

Cite the bracket labels exactly (C1, R2). Every claim must cite the hits it relies on.
A claim with no citation is speculation. Do not invent rules or cards that are not in the hits."""


def describe(card: CardInput) -> str:
    details: list[str] = []
    if card.power or card.toughness:
        details.append(f"power/toughness {card.power}/{card.toughness}")
    if card.loyalty:
        details.append(f"loyalty {card.loyalty}")
    if card.cmc is not None:
        details.append(f"mana value {card.cmc:g}")
    if card.colors:
        details.append("colors " + " ".join(card.colors))
    if card.keywords:
        details.append("keywords " + ", ".join(card.keywords))
    text = card.as_text()
    return text if not details else f"{text}\n" + "; ".join(details)


def _numbered(hits: list[Hit]) -> tuple[dict[str, str], str]:
    labels: dict[str, str] = {}
    lines: list[str] = []
    cards = rules = 0
    for hit in hits:
        if hit.namespace == "rules":
            rules += 1
            label = f"R{rules}"
        else:
            cards += 1
            label = f"C{cards}"
        labels[label] = hit.id
        labels[label.casefold()] = hit.id
        lines.append(f"[{label}] {hit.text}")
    return labels, "\n".join(lines) if lines else "(no hits)"


def _remap(citations: list[str], labels: dict[str, str]) -> list[str]:
    remapped: list[str] = []
    for citation in citations:
        token = citation.strip().strip("[]")
        remapped.append(labels.get(token) or labels.get(token.casefold()) or token)
    return remapped


@register("generate", "llm_judge")
class LlmJudge:
    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self.structured = llm.with_structured_output(BalanceReport)

    def generate(self, card: CardInput, hits: list[Hit]) -> BalanceReport:
        labels, numbered = _numbered(hits)
        prompt = f"Card:\n{describe(card)}\n\nHits:\n{numbered}"
        result = self.structured.invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=prompt),
        ])
        report = result if isinstance(result, BalanceReport) else BalanceReport.model_validate(result)
        data = report.model_dump()
        for issue in data["wording_issues"]:
            issue["rule_ids"] = _remap(issue["rule_ids"], labels)
        for combo in data["combos"]:
            combo["card_ids"] = _remap(combo["card_ids"], labels)
        data["power_assessment"]["comparable_ids"] = _remap(
            data["power_assessment"]["comparable_ids"], labels
        )
        return BalanceReport.model_validate(data)
