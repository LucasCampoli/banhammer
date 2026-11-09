"""What one card evaluation carries from node to node."""
from __future__ import annotations

from typing import TypedDict

from banhammer.models import BalanceReport, CardInput, Hit, Query


class PipelineState(TypedDict, total=False):
    card: CardInput
    format: str               # "modern" or "standard", the format the card is audited against
    queries: list[Query]
    hits: list[Hit]
    report: BalanceReport
    dropped_claims: int       # claims removed from the report for lacking a valid citation
