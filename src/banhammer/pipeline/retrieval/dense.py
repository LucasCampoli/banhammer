"""Similarity search in one namespace."""
from __future__ import annotations

from typing import Any

from langchain_core.vectorstores import InMemoryVectorStore

from banhammer.models import Hit, Query
from banhammer.shared.registry import register


def _filter_for(store: Any, metadata_filter: dict | None) -> Any:
    """Pinecone takes a dict. The in-memory store takes a function."""
    if not metadata_filter:
        return None
    if not isinstance(store, InMemoryVectorStore):
        return metadata_filter

    def matches(doc: Any, wanted: dict = metadata_filter) -> bool:
        return all(doc.metadata.get(key) == value for key, value in wanted.items())

    return matches


@register("retrieve", "dense")
class Dense:
    def __init__(self, stores: Any, k: int = 10) -> None:
        self.stores = stores
        self.k = int(k)

    def search(self, query: Query, k: int | None = None) -> list[Hit]:
        store = self.stores.get(query.namespace)
        kwargs = {}
        metadata_filter = _filter_for(store, query.filter)
        if metadata_filter is not None:
            kwargs["filter"] = metadata_filter
        pairs = store.similarity_search_with_score(query.text, k=self.k if k is None else k, **kwargs)
        hits = []
        for doc, score in pairs:
            metadata = dict(doc.metadata or {})
            metadata.setdefault("text", doc.page_content)
            hit_id = doc.id or metadata.get("oracle_id") or metadata.get("rule") or ""
            hits.append(Hit(
                id=str(hit_id),
                namespace=query.namespace,
                text=doc.page_content,
                metadata=metadata,
                score=float(score),
                source="dense",
            ))
        return hits
