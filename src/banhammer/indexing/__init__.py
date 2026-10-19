"""Read processed JSONL and upsert it. Indexing never downloads."""

from banhammer.indexing.run import index_processed

__all__ = ["index_processed"]
