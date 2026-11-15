"""OpenAI embeddings. The model must match the one used to build the index."""
from __future__ import annotations

from langchain_openai import OpenAIEmbeddings


def get_embeddings(cfg: dict) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=cfg["model"])
