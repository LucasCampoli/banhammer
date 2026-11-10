"""Retrieval helpers: catalog filters, BM25 over the catalog and reciprocal rank fusion."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from banhammer.models import Hit, Query
from banhammer.shared.documents import read_jsonl

RRF_C = 60
_TOKEN = re.compile(r"[a-z0-9]+")


def format_filter(fmt: str) -> dict:
    """Pinecone filter for catalog cards legal or banned in the audited format."""
    return {"formats": {"$in": [fmt]}}


def matches(metadata: dict[str, Any], flt: dict[str, Any] | None) -> bool:
    """Evaluate a Pinecone metadata filter in Python.

    Supports ``$and``, ``$in`` (a list field matches when any of its items is in the list) and
    ``$lte``. A missing field never matches a condition.

    Raises:
        ValueError: If the filter uses another operator.
    """
    if not flt:
        return True
    for key, condition in flt.items():
        if key == "$and":
            if not all(matches(metadata, clause) for clause in condition):
                return False
            continue
        value = metadata.get(key)
        for op, arg in condition.items():
            if op == "$in":
                values = value if isinstance(value, list) else [value]
                if not any(item in arg for item in values if item is not None):
                    return False
            elif op == "$lte":
                if value is None or float(value) > float(arg):
                    return False
            else:
                raise ValueError(f"unsupported filter operator {op!r}")
    return True


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, for BM25."""
    return _TOKEN.findall(text.lower())


class BM25Index:
    """Lexical BM25 index over the processed catalog (``cards.jsonl``), built in memory."""

    def __init__(self, path: Path | str):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `python -m banhammer.ingest` first (hybrid: true)")
        self.docs = read_jsonl(path)
        if not self.docs:
            raise ValueError(f"no cards in {path}")
        self.bm25 = BM25Okapi([tokenize(doc.page_content) for doc in self.docs])

    def search(self, query: Query, k: int) -> list[Hit]:
        """Top-k cards by BM25 score that pass ``query.filter``. Cards with score 0 are skipped."""
        tokens = tokenize(query.text)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        hits: list[Hit] = []
        for index in np.argsort(-scores, kind="stable"):
            if len(hits) == k or scores[index] <= 0:
                break
            doc = self.docs[index]
            if not matches(doc.metadata, query.filter):
                continue
            metadata = dict(doc.metadata)
            metadata.setdefault("text", doc.page_content)
            hits.append(Hit(id=str(doc.id), namespace="cards", text=doc.page_content,
                            metadata=metadata, score=float(scores[index]), source="bm25"))
        return hits


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
