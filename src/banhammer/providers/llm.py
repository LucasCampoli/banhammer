"""OpenAI chat model used to write the balance report."""
from __future__ import annotations

from typing import Any


def get_llm(cfg: dict) -> Any:
    provider = cfg.get("provider", "openai")
    if provider != "openai":
        raise ValueError(f"llm provider must be 'openai', got {provider!r}")
    from langchain_openai import ChatOpenAI

    kwargs = {"model": cfg.get("model", "gpt-4o-mini")}
    if cfg.get("temperature") is not None:
        kwargs["temperature"] = cfg["temperature"]
    return ChatOpenAI(**kwargs)
