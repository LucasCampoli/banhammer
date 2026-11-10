"""What one card evaluation carries from node to node."""
from __future__ import annotations

from typing import TypedDict

from banhammer.models import BalanceReport, CardAnalysis, CardInput, Hit, Query


class PipelineState(TypedDict, total=False):
    card: CardInput
    format: str               # "modern" or "standard", the format the card is audited against
    analysis: CardAnalysis | None  # decomposition behind the expanded queries (decompose: true)
    queries: list[Query]
    hits: list[Hit]
    report: BalanceReport
    dropped_claims: int       # claims removed from the report for lacking a valid citation
