"""OpenAI chat model used to write the balance report."""
from __future__ import annotations

from langchain_openai import ChatOpenAI


def get_llm(cfg: dict) -> ChatOpenAI:
    return ChatOpenAI(model=cfg["model"], temperature=cfg["temperature"])
