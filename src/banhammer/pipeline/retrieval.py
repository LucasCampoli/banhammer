"""Retrieval helpers: catalog filters and reciprocal rank fusion."""
from __future__ import annotations

from banhammer.models import Hit

RRF_C = 60


def format_filter(fmt: str) -> dict:
    """Pinecone filter for catalog cards legal or banned in the audited format."""
    return {"formats": {"$in": [fmt]}}


def rrf(lists: list[list[Hit]], c: int = RRF_C) -> list[Hit]:
    """Fuse ranked lists with reciprocal rank fusion.

    Each list adds ``1 / (c + rank)`` (rank starts at 1) to every id it contains. Duplicates are
    merged into one hit that keeps the first copy seen, the fused score and the union of the
    sources (``"bm25+dense"``). Ties keep the order in which the ids were first seen.
    """
    scores: dict[str, float] = {}
    first: dict[str, Hit] = {}
    sources: dict[str, set[str]] = {}
    for hits in lists:
        for rank, hit in enumerate(hits, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (c + rank)
            first.setdefault(hit.id, hit)
            sources.setdefault(hit.id, set()).update(hit.source.split("+"))
    ordered = sorted(scores, key=scores.__getitem__, reverse=True)
    return [
        first[hit_id].model_copy(update={"score": scores[hit_id], "source": "+".join(sorted(sources[hit_id]))})
        for hit_id in ordered
    ]


def fuse_by_namespace(lists: dict[str, list[list[Hit]]]) -> list[Hit]:
    """One ranked list per namespace, in the order the namespaces were added.

    A namespace with a single list is returned untouched, so a pipeline with one query per
    namespace (version 1) keeps its hits and scores exactly as Pinecone returned them.
    """
    fused: list[Hit] = []
    for namespace_lists in lists.values():
        fused.extend(namespace_lists[0] if len(namespace_lists) == 1 else rrf(namespace_lists))
    return fused
