from banhammer.ocr.pipeline import clean_text, split_pt


def test_split_pt_and_clean_text():
    assert split_pt("2/5") == ("2", "5")
    assert split_pt("garbage") == (None, None)
    assert clean_text("a   b\n c ") == "a b c"
    assert clean_text("a  b\n\n c", multiline=True) == "a b\nc"
