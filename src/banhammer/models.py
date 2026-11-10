"""Pydantic models shared across pipeline stages."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from banhammer.shared.cardtext import colors_in, format_card, mana_value


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


Format = Literal["modern", "standard"]
FORMATS: tuple[str, ...] = ("modern", "standard")


class OCRCard(BaseModel):
    name: str = ""
    mana_cost: str = ""
    type_line: str = ""
    rules_text: str = ""
    power: str | None = None
    toughness: str | None = None
    confidence: dict[str, float | None] = Field(default_factory=dict)


def _blank_to_none(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


class CardFace(BaseModel):
    """One face of a double-faced, split or adventure card."""

    model_config = ConfigDict(extra="ignore")

    name: str = ""
    mana_cost: str = ""
    type_line: str = ""
    oracle_text: str = ""
    power: str | None = None
    toughness: str | None = None
    loyalty: str | None = None

    @field_validator("power", "toughness", "loyalty", mode="before")
    @classmethod
    def _stats(cls, value: Any) -> str | None:
        return _blank_to_none(value)


class CardInput(BaseModel):
    """Card to evaluate. Extra Scryfall fields are ignored.

    cmc and colors are filled from the mana cost when the JSON leaves them out.
    If the card itself has no mana cost, the face costs are added together.
    """

    model_config = ConfigDict(extra="ignore")

    name: str
    mana_cost: str = ""
    type_line: str = ""
    oracle_text: str = ""
    power: str | None = None
    toughness: str | None = None
    loyalty: str | None = None
    card_faces: list[CardFace] = Field(default_factory=list)
    cmc: float | None = None
    colors: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    oracle_id: str | None = None

    @field_validator("power", "toughness", "loyalty", mode="before")
    @classmethod
    def _stats(cls, value: Any) -> str | None:
        return _blank_to_none(value)

    @field_validator("mana_cost", "type_line", "oracle_text", mode="before")
    @classmethod
    def _text(cls, value: Any) -> str:
        return "" if value is None else str(value)

    @field_validator("colors", "keywords", mode="before")
    @classmethod
    def _lists(cls, value: Any) -> list:
        return [] if value is None else value

    @model_validator(mode="before")
    @classmethod
    def _derive_missing(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        faces = data.get("card_faces") or []
        mana = data.get("mana_cost") or ""
        if data.get("cmc") is None:
            if mana:
                data["cmc"] = mana_value(str(mana))
            else:
                data["cmc"] = sum(mana_value(str((face or {}).get("mana_cost") or "")) for face in faces)
        if "colors" not in data or data["colors"] is None:
            blob = str(mana) + "".join(str((face or {}).get("mana_cost") or "") for face in faces)
            data["colors"] = colors_in(blob)
        if data.get("keywords") is None:
            data["keywords"] = []
        return data

    def names(self) -> set[str]:
        found = {self.name.casefold()} if self.name else set()
        found.update(face.name.casefold() for face in self.card_faces if face.name)
        return found

    def as_text(self) -> str:
        faces = [face.model_dump() for face in self.card_faces] or None
        return format_card(self.name, self.type_line, self.mana_cost, self.oracle_text, faces)


class Query(BaseModel):
    text: str
    namespace: Literal["cards", "rules"]
    filter: dict[str, Any] | None = None


class Hit(BaseModel):
    id: str
    namespace: Literal["cards", "rules"]
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None
    source: str


class Ability(BaseModel):
    """One ability of the evaluated card, split into functional primitives."""

    text: str = Field(description="The ability exactly as written on the card.")
    trigger: str = Field("", description="What makes it happen (e.g. 'whenever an opponent draws a card'). "
                                         "Empty for static and activated abilities.")
    cost: str = Field("", description="What must be paid to activate it (e.g. '{1}, {T}, sacrifice a creature'). "
                                      "Empty if it has no cost.")
    effect: str = Field(description="What the ability does when it resolves or applies.")
    restrictions: str = Field("", description="Limits on when or how often it works "
                                              "(e.g. 'only as a sorcery', 'once each turn'). Empty if none.")


class Counterpart(BaseModel):
    """The other half of a potential combo, described as the card it would take."""

    description: str = Field(description="Oracle-style English rules text of an existing card that would complete "
                                         "a combo with the evaluated card (the other half of the combo, not a "
                                         "similar card).")
    card_type: str | None = Field(None, description="One card type word the counterpart must have: Creature, "
                                                    "Artifact, Enchantment, Land, Instant, Sorcery or "
                                                    "Planeswalker. Null if any type works.")
    max_mana_value: int | None = Field(None, description="Highest mana value that keeps the combo practical. "
                                                         "Null if it does not matter.")


class CardAnalysis(BaseModel):
    """Decomposition of the evaluated card used to expand the retrieval queries."""

    abilities: list[Ability] = Field(default_factory=list, description="Every ability of the card.")
    counterparts: list[Counterpart] = Field(default_factory=list,
                                            description="Up to 5 mechanical counterparts that would combo with it.")
    rules_topics: list[str] = Field(default_factory=list,
                                    description="Up to 5 Comprehensive Rules topics its wording depends on "
                                                "(keywords, token types, timing, replacement effects).")


class WordingIssue(BaseModel):
    description: str
    rule_ids: list[str] = Field(default_factory=list)


class Combo(BaseModel):
    description: str
    card_ids: list[str] = Field(default_factory=list)
    sequence: list[str] = Field(default_factory=list)


class PowerAssessment(BaseModel):
    summary: str = ""
    comparable_ids: list[str] = Field(default_factory=list)


class BalanceReport(BaseModel):
    wording_issues: list[WordingIssue] = Field(default_factory=list)
    combos: list[Combo] = Field(default_factory=list)
    power_assessment: PowerAssessment = Field(default_factory=PowerAssessment)
    risk: Literal["ok", "watch", "likely_ban"]
    confidence: float = Field(ge=0, le=1)
