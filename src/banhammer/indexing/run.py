"""Embed `data/processed/*.jsonl` into one index with a `cards` and a `rules` namespace."""
from __future__ import annotations

import argparse
from pathlib import Path

from banhammer.shared.config import ROOT, load_env, load_yaml
from banhammer.shared.documents import read_jsonl
from banhammer.providers import StoreFactory, get_embeddings

NAMESPACES = ("cards", "rules")


def index_processed(cfg: dict) -> dict[str, int]:
    stores = StoreFactory(get_embeddings(cfg["embeddings"]), model=cfg["embeddings"]["model"])
    print(f"index: {stores.ensure_index(cfg['index'])}")
    processed = ROOT / cfg["processed_dir"]
    batch_size = cfg["index"]["batch_size"]
    counts: dict[str, int] = {}
    for namespace in NAMESPACES:
        docs = read_jsonl(processed / f"{namespace}.jsonl")
        store = stores.get(namespace)
        for start in range(0, len(docs), batch_size):
            batch = docs[start:start + batch_size]
            store.add_documents(batch, ids=[doc.id for doc in batch], async_req=False)
            print(f"{namespace}: {min(start + batch_size, len(docs))}/{len(docs)}")
        counts[namespace] = len(docs)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m banhammer.index",
        description="Embed processed JSONL into the vector store. Does not download anything.",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "ingest.yaml")
    args = parser.parse_args()
    load_env()
    counts = index_processed(load_yaml(args.config))
    for namespace, count in counts.items():
        print(f"{namespace}: indexed {count} records")


if __name__ == "__main__":
    main()
