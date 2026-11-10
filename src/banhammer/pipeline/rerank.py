"""Rerank the fused catalog candidates before the judge sees them.

Two rerankers are compared in the ablation: an LLM that scores how likely each candidate is to
combo with the card, and a pretrained cross-encoder. There is no labelled dataset of broken
interactions to fine-tune a cross-encoder on, so it is used as published.
"""
from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from banhammer.models import CardInput, Hit

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_SCORE = 10.0

_RERANK_SYSTEM = """You are a Magic: The Gathering combo expert.
You are given a newly designed card and candidate existing cards labeled [C1], [C2], ...
For every candidate, score from 0 to 10 how likely it is that the candidate and the evaluated
card together form an infinite loop, a lock or a degenerate engine. 0 means no meaningful
interaction, 10 means a known or obvious game-winning combo. Score every label exactly once."""


class CandidateScore(BaseModel):
    label: str = Field(description="The candidate label, e.g. C3.")
    score: float = Field(description="0 to 10: how likely the candidate combos with the evaluated card.")


class CandidateScores(BaseModel):
    scores: list[CandidateScore] = Field(default_factory=list)


def _by_score(hits: list[Hit], scores: list[float]) -> list[Hit]:
    """Hits sorted by score, highest first. Ties keep their incoming (RRF) order."""
    order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
    return [hits[i].model_copy(update={"score": scores[i]}) for i in order]


def llm_rerank(llm: Any, card: CardInput, hits: list[Hit]) -> list[Hit]:
    """Reorder candidates by an LLM combo score, in a single structured call.

    A label the LLM leaves out scores 0; scores are clamped to [0, 10].
    """
    if not hits:
        return []
    candidates = "\n".join(f"[C{i}] {hit.text}" for i, hit in enumerate(hits, start=1))
    result = llm.with_structured_output(CandidateScores).invoke([
        SystemMessage(content=_RERANK_SYSTEM),
        HumanMessage(content=f"Evaluated card:\n{card.as_text()}\n\nCandidates:\n{candidates}"),
    ])
    parsed = result if isinstance(result, CandidateScores) else CandidateScores.model_validate(result)
    by_label = {item.label.strip().strip("[]").upper(): item.score for item in parsed.scores}
    scores = [min(max(by_label.get(f"C{i}", 0.0), 0.0), MAX_SCORE) for i in range(1, len(hits) + 1)]
    return _by_score(hits, scores)


@lru_cache(maxsize=1)
def _cross_encoder(model_name: str) -> Any:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "rerank: cross_encoder needs sentence-transformers; run `pip install -e \".[rerank]\"`"
        ) from exc
    return CrossEncoder(model_name)


def cross_encoder_rerank(card: CardInput, hits: list[Hit], model_name: str = CROSS_ENCODER_MODEL) -> list[Hit]:
    """Reorder candidates by a pretrained cross-encoder score of (card text, candidate text)."""
    if not hits:
        return []
    model = _cross_encoder(model_name)
    scores = model.predict([(card.as_text(), hit.text) for hit in hits])
    return _by_score(hits, [float(score) for score in scores])


def rerank_cards(hits: list[Hit], reorder: Callable[[list[Hit]], list[Hit]], candidates: int) -> list[Hit]:
    """Rerank the first ``candidates`` card hits; the other cards follow, then the rules."""
    cards = [hit for hit in hits if hit.namespace == "cards"]
    rules = [hit for hit in hits if hit.namespace != "cards"]
    return reorder(cards[:candidates]) + cards[candidates:] + rules
