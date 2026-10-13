import pytest
from pydantic import ValidationError

from banhammer.models import OCRCard, ZoneBox


def test_ocrcard_defaults_and_json_roundtrip():
    card = OCRCard(name="Abrade", mana_cost="{1}{R}", confidence={"name": 0.9, "mana_cost": None})
    assert card.power is None and card.rules_text == ""
    assert OCRCard.model_validate_json(card.model_dump_json()) == card


def test_zonebox_validation():
    with pytest.raises(ValidationError):
        ZoneBox(x0=0.5, y0=0.1, x1=0.4, y1=0.2)
    with pytest.raises(ValidationError):
        ZoneBox(x0=0.0, y0=0.0, x1=1.5, y1=0.2)


def test_zonebox_to_pixels():
    assert ZoneBox(x0=0.1, y0=0.5, x1=0.5, y1=1.0).to_pixels(672, 936) == (67, 468, 336, 936)
