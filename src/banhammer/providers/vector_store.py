"""Pinecone index, one namespace for cards and one for rules."""
from __future__ import annotations

import os
import re
import time

from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


def index_name_for(model: str) -> str:
    """Pinecone index names cannot be shared across embedding models."""
    slug = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")
    name = f"banhammer-{slug}"
    return name[:45].rstrip("-")


class StoreFactory:
    """One store per namespace of the index built for this embedding model."""

    def __init__(self, embeddings: Embeddings, model: str) -> None:
        self.embeddings = embeddings
        self.index_name = index_name_for(model)
        self._stores: dict[str, PineconeVectorStore] = {}

    def ensure_index(self, index_cfg: dict) -> str:
        """Create the Pinecone index when it is missing. `index_cfg` is the `index` block of ingest.yaml."""
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY is not set")
        client = Pinecone(api_key=api_key)
        if self.index_name not in client.list_indexes().names():
            client.create_index(
                name=self.index_name,
                dimension=index_cfg["dimension"],
                metric=index_cfg["metric"],
                spec=ServerlessSpec(cloud=index_cfg["cloud"], region=index_cfg["region"]),
            )
        for _ in range(60):
            if client.describe_index(self.index_name).status.ready:
                return self.index_name
            time.sleep(2)
        raise TimeoutError(f"Pinecone index {self.index_name} was not ready")

    def get(self, namespace: str) -> PineconeVectorStore:
        if namespace not in self._stores:
            self._stores[namespace] = PineconeVectorStore(
                index_name=self.index_name,
                embedding=self.embeddings,
                namespace=namespace,
            )
        return self._stores[namespace]
