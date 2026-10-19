"""What one card evaluation carries from node to node."""
from __future__ import annotations

from typing import TypedDict

from banhammer.models import BalanceReport, CardInput, Hit, Query


class PipelineState(TypedDict, total=False):
    card: CardInput
    queries: list[Query]
    hits: list[Hit]
    report: BalanceReport
    # One list per retriever, only when more than one retriever is enabled.
    hit_lists: list[list[Hit]]
