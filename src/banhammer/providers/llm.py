"""OpenAI chat model used to write the balance report."""
from __future__ import annotations

from langchain_openai import ChatOpenAI

# Reasoning models (gpt-5 and later) reject `temperature` and take `reasoning_effort` instead.
OPTIONAL_PARAMS = ("temperature", "reasoning_effort")


def get_llm(cfg: dict) -> ChatOpenAI:
    params = {name: cfg[name] for name in OPTIONAL_PARAMS if cfg.get(name) is not None}
    return ChatOpenAI(model=cfg["model"], **params)
