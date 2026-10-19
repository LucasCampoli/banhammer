"""Vector store factory: pinecone or an in-memory store for tests."""
from __future__ import annotations

import os
import re
import time
from typing import Any


def index_name_for(model: str) -> str:
    """Pinecone index names cannot be shared across embedding models."""
    slug = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")
    name = f"banhammer-{slug}"
    return name[:45].rstrip("-")


def _index_ready(description: Any) -> bool:
    status = description.status
    if isinstance(status, dict):
        return bool(status.get("ready"))
    return bool(getattr(status, "ready", False))


class StoreFactory:
    """One store per namespace, built from the vector_store config block."""

    def __init__(self, store_cfg: dict, embeddings: Any, model: str, index_cfg: dict | None = None) -> None:
        self.store_cfg = store_cfg
        self.embeddings = embeddings
        self.model = model
        self.index_name = index_name_for(model)
        self.index_cfg = index_cfg or {}
        self._stores: dict[str, Any] = {}

    def ensure_index(self) -> str:
        """Create the Pinecone index when it is missing. No-op for the memory provider."""
        if self.store_cfg["provider"] != "pinecone":
            return self.index_name
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY is not set")
        from pinecone import Pinecone, ServerlessSpec

        client = Pinecone(api_key=api_key)
        if self.index_name not in client.list_indexes().names():
            client.create_index(
                name=self.index_name,
                dimension=int(self.index_cfg.get("dimension", 1536)),
                metric=self.index_cfg.get("metric", "cosine"),
                spec=ServerlessSpec(
                    cloud=self.index_cfg.get("cloud", "aws"),
                    region=self.index_cfg.get("region", "us-east-1"),
                ),
            )
        for _ in range(60):
            if _index_ready(client.describe_index(self.index_name)):
                return self.index_name
            time.sleep(2)
        raise TimeoutError(f"Pinecone index {self.index_name} was not ready")

    def get(self, namespace: str) -> Any:
        if namespace in self._stores:
            return self._stores[namespace]
        provider = self.store_cfg["provider"]
        if provider == "memory":
            from langchain_core.vectorstores import InMemoryVectorStore

            store = InMemoryVectorStore(embedding=self.embeddings)
        elif provider == "pinecone":
            from langchain_pinecone import PineconeVectorStore

            store = PineconeVectorStore(
                index_name=self.index_name,
                embedding=self.embeddings,
                namespace=namespace,
            )
        else:
            raise ValueError(f"unknown vector_store provider {provider!r}")
        self._stores[namespace] = store
        return store
