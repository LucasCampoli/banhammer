"""JSONL records exchanged by ingestion and indexing."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


def clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Drop nulls and empty lists so the record can be stored in Pinecone."""
    cleaned: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, list):
            items = [str(item) for item in value if item is not None]
            if items:
                cleaned[key] = items
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
            continue
        cleaned[key] = str(value)
    return cleaned


def write_jsonl(path: Path, docs: list[Document]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for doc in docs:
            row = {"id": doc.id, "text": doc.page_content, "metadata": doc.metadata}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[Document]:
    docs: list[Document] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            docs.append(Document(id=row["id"], page_content=row["text"], metadata=row.get("metadata") or {}))
    return docs
