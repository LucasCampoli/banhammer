from banhammer.ocr.pipeline import clean_text, split_pt, strip_edge_chars, line_text
from banhammer.ocr.tesseract import ReadResult, Word


def test_split_pt_and_clean_text():
    assert split_pt("2/5") == ("2", "5")
    assert split_pt("garbage") == (None, None)
    assert clean_text("a   b\n c ") == "a b c"
    assert clean_text("a  b\n\n c", multiline=True) == "a b\nc"


def test_strip_edge_chars_keeps_inner_and_allowed_punctuation():
    assert strip_edge_chars("| Creature — Human Wizard ~~") == "Creature — Human Wizard"
    assert strip_edge_chars("'Ent's Fury") == "Ent's Fury"
    assert strip_edge_chars("~~") == ""
    assert strip_edge_chars("(+2 Mace", keep_leading="+") == "+2 Mace"
    assert strip_edge_chars("Shields Up! |", keep_trailing="!.") == "Shields Up!"


def test_line_text_drops_set_symbol_and_detached_marks():
    words = [Word("|", 0.9, 0, 0, 6, 40), Word("Double", 0.96, 40, 0, 100, 40),
             Word("Cleave", 0.9, 160, 0, 100, 40), Word("oe", 0.0, 420, 0, 20, 40), Word("S", 0.8, 600, 0, 20, 40)]
    result = ReadResult("", None, [words])
    assert line_text(result, scale=2.0, max_gap=30, right_limit=290) == "Double Cleave"
