"""Token usage and dollar cost of a pipeline run.

Chat usage comes from LangChain's usage callback, keyed by the model name the API
reports (e.g. ``gpt-4o-mini-2024-07-18``). Embedding calls do not report usage
through callbacks, so their tokens are counted locally with tiktoken.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import tiktoken

# USD per 1M tokens: (input, cached input, output). OpenAI list prices; check them
# against https://openai.com/api/pricing before relying on the totals.
PRICES: dict[str, tuple[float, float, float]] = {
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-4o": (2.50, 1.25, 10.00),
    "text-embedding-3-small": (0.02, 0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.13, 0.0),
}

_ENCODING = tiktoken.get_encoding("cl100k_base")  # the encoding of the text-embedding-3 models


def price_for(model: str) -> tuple[float, float, float] | None:
    """Prices for a model name, matching dated snapshots by their longest known prefix."""
    matches = [name for name in PRICES if model == name or model.startswith(f"{name}-")]
    return PRICES[max(matches, key=len)] if matches else None


def chat_usage(usage: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Sum the callback's per-model usage into tokens and cost. Cost is None for an unknown model."""
    totals = {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0}
    cost: float | None = 0.0
    for model, meta in usage.items():
        inputs = int(meta.get("input_tokens", 0))
        cached = int((meta.get("input_token_details") or {}).get("cache_read", 0) or 0)
        outputs = int(meta.get("output_tokens", 0))
        totals["input_tokens"] += inputs
        totals["cached_tokens"] += cached
        totals["output_tokens"] += outputs
        prices = price_for(model)
        if prices is None or cost is None:
            cost = None
            continue
        price_in, price_cached, price_out = prices
        cost += ((inputs - cached) * price_in + cached * price_cached + outputs * price_out) / 1_000_000
    return {**totals, "chat_cost_usd": cost}


def embedding_tokens(texts: Iterable[str]) -> int:
    return sum(len(_ENCODING.encode(text)) for text in texts)


def embedding_cost(tokens: int, model: str) -> float | None:
    prices = price_for(model)
    return None if prices is None else tokens * prices[0] / 1_000_000
