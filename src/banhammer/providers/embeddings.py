"""OpenAI embeddings. The model must match the one used to build the index."""
from __future__ import annotations

from typing import Any


def get_embeddings(cfg: dict) -> Any:
    provider = cfg.get("provider", "openai")
    if provider != "openai":
        raise ValueError(f"embeddings provider must be 'openai', got {provider!r}")
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=cfg.get("model", "text-embedding-3-small"))
