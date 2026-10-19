"""Embedding factory: openai, huggingface, fake."""
from __future__ import annotations

from typing import Any


def get_embeddings(cfg: dict) -> Any:
    provider = cfg["provider"]
    if provider == "fake":
        from langchain_core.embeddings import DeterministicFakeEmbedding

        return DeterministicFakeEmbedding(size=int(cfg.get("size", 32)))
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=cfg.get("model", "text-embedding-3-small"))
    if provider == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise RuntimeError(
                "huggingface embeddings require the hf extra: pip install -e '.[hf]'"
            ) from exc
        return HuggingFaceEmbeddings(model_name=cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2"))
    raise ValueError(f"unknown embeddings provider {provider!r}")
