"""Pydantic models shared across pipeline stages."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ZoneBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float

    @field_validator("x0", "y0", "x1", "y1")
    @classmethod
    def _unit(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("zone coordinates must be within [0, 1]")
        return v

    def model_post_init(self, _ctx) -> None:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("zone must have x1 > x0 and y1 > y0")

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        return (round(self.x0 * width), round(self.y0 * height),
                round(self.x1 * width), round(self.y1 * height))


class OCRCard(BaseModel):
    name: str = ""
    mana_cost: str = ""
    type_line: str = ""
    rules_text: str = ""
    power: str | None = None
    toughness: str | None = None
    confidence: dict[str, float | None] = Field(default_factory=dict)
