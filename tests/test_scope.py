from banhammer.ingest.scryfall_cards import is_ingested
from banhammer.ocr.scope import in_ocr_scope

CREATURE = {
    "layout": "normal",
    "frame": "2015",
    "border_color": "black",
    "frame_effects": ["legendary"],
    "type_line": "Legendary Creature — Avatar",
    "legalities": {"modern": "legal"},
    "oracle_id": "x",
}


def test_normal_2003_and_2015_cards_are_in_scope():
    assert in_ocr_scope(CREATURE)
    assert in_ocr_scope({**CREATURE, "frame": "2003"})


def test_other_layouts_frames_and_planeswalkers_are_out_of_scope():
    assert not in_ocr_scope({**CREATURE, "layout": "transform"})
    assert not in_ocr_scope({**CREATURE, "frame": "1997"})
    assert not in_ocr_scope({**CREATURE, "type_line": "Legendary Planeswalker — Karn"})
    assert not in_ocr_scope({**CREATURE, "border_color": "borderless"})
    assert not in_ocr_scope({**CREATURE, "frame_effects": ["extendedart"]})


def test_ingested_population_is_modern_legal_or_banned():
    assert is_ingested(CREATURE)
    assert is_ingested({**CREATURE, "legalities": {"modern": "banned"}})
    assert not is_ingested({**CREATURE, "legalities": {"modern": "not_legal"}})
    assert not is_ingested({**CREATURE, "layout": "token"})
    assert not is_ingested({**CREATURE, "oracle_id": None})
