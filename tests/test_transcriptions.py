import io
import json

import pytest

from banhammer.ingest.__main__ import card_source
from banhammer.ingest.scryfall_cards import ScryfallCards, card_to_document, transcription_for
from banhammer.models import OCRCard
from banhammer.ocr.transcriptions import is_usable, read_transcriptions, transcribed_ids, write_transcription

CARD = {
    "id": "printing-1",
    "oracle_id": "oracle-1",
    "name": "Ghitu Journeymage",
    "layout": "normal",
    "frame": "2015",
    "type_line": "Creature — Human Wizard",
    "mana_cost": "{2}{R}",
    "oracle_text": "When this creature enters, if you control another Wizard, this creature deals 2 damage.",
    "cmc": 3.0,
    "colors": ["R"],
    "legalities": {"modern": "legal"},
}
READING = OCRCard(
    name="Ghitu Journeymage",
    mana_cost="{2}{R}",
    type_line="Creature — Human Wizard",
    rules_text="When Ghitu Journeymage enters the battlefield, if you control another Wizard, ...",
)


def test_a_transcription_replaces_the_text_but_not_the_metadata():
    doc = card_to_document(CARD, READING)
    assert "enters the battlefield" in doc.page_content
    assert doc.metadata["source"] == "ocr"
    assert doc.metadata["name"] == "Ghitu Journeymage"
    assert doc.metadata["mana_cost"] == "{2}{R}"


def test_without_a_transcription_the_card_keeps_its_scryfall_text():
    doc = card_to_document(CARD)
    assert "this creature enters" in doc.page_content
    assert doc.metadata["source"] == "scryfall"


def test_transcription_must_come_from_the_same_printing_of_a_card_in_scope():
    transcriptions = {"oracle-1": ("printing-1", READING)}
    assert transcription_for(CARD, transcriptions) is READING
    assert transcription_for({**CARD, "id": "printing-2"}, transcriptions) is None
    assert transcription_for({**CARD, "layout": "transform"}, transcriptions) is None
    assert transcription_for({**CARD, "oracle_id": "oracle-2"}, transcriptions) is None


def test_a_reading_without_name_or_type_line_is_not_usable():
    assert is_usable(READING)
    assert not is_usable(READING.model_copy(update={"name": " "}))
    assert not is_usable(READING.model_copy(update={"type_line": ""}))
    assert is_usable(READING.model_copy(update={"rules_text": ""}))  # vanilla creature


def test_transcriptions_file_keeps_failures_for_resuming_but_reads_only_usable_cards(tmp_path):
    path = tmp_path / "ocr_cards.jsonl"
    buffer = io.StringIO()
    write_transcription(buffer, "printing-1", "oracle-1", READING)
    write_transcription(buffer, "printing-2", "oracle-2", OSError("image not found"))
    write_transcription(buffer, "printing-3", "oracle-3", READING.model_copy(update={"name": ""}))
    path.write_text(buffer.getvalue(), encoding="utf-8")

    assert transcribed_ids(path) == {"printing-1", "printing-2", "printing-3"}
    assert transcribed_ids(tmp_path / "missing.jsonl") == set()
    assert read_transcriptions(path) == {"oracle-1": ("printing-1", READING)}


def _bulk_and_transcriptions(tmp_path):
    bulk = tmp_path / "oracle-cards.jsonl"
    bulk.write_text(json.dumps(CARD) + "\n", encoding="utf-8")
    transcriptions = tmp_path / "ocr_cards.jsonl"
    with transcriptions.open("w", encoding="utf-8") as handle:
        write_transcription(handle, CARD["id"], CARD["oracle_id"], READING)
    return bulk, transcriptions


def test_card_text_can_come_from_the_ocr_or_only_from_scryfall(tmp_path):
    bulk, transcriptions = _bulk_and_transcriptions(tmp_path)
    [ocr] = ScryfallCards(transcriptions).normalize(bulk)
    [scryfall] = ScryfallCards(transcriptions=None).normalize(bulk)
    assert ocr.metadata["source"] == "ocr"
    assert scryfall.metadata["source"] == "scryfall"
    assert card_source("scryfall").transcriptions is None


def test_asking_for_ocr_text_without_transcriptions_is_an_error(tmp_path):
    bulk, _ = _bulk_and_transcriptions(tmp_path)
    missing = tmp_path / "missing.jsonl"
    [doc] = ScryfallCards(missing).normalize(bulk)  # default: fall back to Scryfall text
    assert doc.metadata["source"] == "scryfall"
    with pytest.raises(FileNotFoundError):
        ScryfallCards(missing, require_transcriptions=True).normalize(bulk)
