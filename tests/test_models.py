import pytest
from pydantic import ValidationError

from banhammer.models import CardInput, OCRCard, ZoneBox


@pytest.mark.parametrize(("layout", "expected"), [("adventure", 3.0), ("modal_dfc", 3.0), ("split", 5.0)])
def test_cardinput_mana_value_of_multi_face_cards(layout, expected):
    faces = [{"name": "Front", "mana_cost": "{2}{R}"}, {"name": "Back", "mana_cost": "{1}{R}"}]
    card = CardInput(name="Front // Back", mana_cost="{2}{R} // {1}{R}", layout=layout, card_faces=faces)
    assert card.cmc == expected


def test_cardinput_mana_value_of_a_single_face_card():
    assert CardInput(name="Bolt", mana_cost="{R}").cmc == 1.0


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
