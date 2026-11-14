from banhammer.ocr.mana import Symbol
from banhammer.ocr.tesseract import Word
from banhammer.ocr.text_symbols import insert_symbols, join_words, restore_symbols


def test_join_words_glues_symbols_and_punctuation():
    assert join_words(["{1}", "{B}", ",", "Discard", "a", "card"]) == "{1}{B}, Discard a card"
    assert join_words(["with", "“", "{T}", ":Add", "one"]) == "with “{T}: Add one"
    assert join_words(["(", "{T}", ")"]) == "({T})"


def test_insert_symbols_places_symbols_on_their_line_or_a_new_one():
    line = [Word("Sacrifice", 0.9, 100, 10, 80, 20)]
    tap = Symbol("T", 0.95, x=10, y=20, diameter=24)  # scale 1: same line as the word
    add = Symbol("C", 0.95, x=60, y=80, diameter=24)  # far below: a line Tesseract did not report
    lines = insert_symbols([line], [tap, add], scale=1.0)
    assert [[w.text for w in ln] for ln in lines] == [["{T}", "Sacrifice"], ["{C}"]]


def test_restore_symbols_replaces_placeholders_only_over_symbols():
    line = [Word("x:", 0.9, 0, 10, 30, 20), Word("Add", 0.9, 40, 10, 50, 20), Word("X", 0.9, 100, 10, 15, 20)]
    tap = Symbol("T", 0.95, x=10, y=20, diameter=24)
    restored = restore_symbols([line], [tap], scale=1.0)
    assert join_words([w.text for w in restored[0]]) == "{T}: Add X"


def test_restore_symbols_drops_doubled_placeholders():
    line = [Word("XxX:", 0.2, 0, 10, 60, 20)]
    symbols = [Symbol("1", 0.9, x=10, y=20, diameter=24), Symbol("R", 0.9, x=40, y=20, diameter=24)]
    restored = restore_symbols([line], symbols, scale=1.0)[0][0]
    assert restored.text == "{1}{R}:" and restored.confidence == 0.9
