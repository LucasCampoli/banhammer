from banhammer.ocr.mana import format_mana_cost


def test_format_mana_cost():
    assert format_mana_cost(["2", "U", "U"]) == "{2}{U}{U}"
    assert format_mana_cost([]) == ""


def test_format_hybrid():
    assert format_mana_cost(["3", "R/G"]) == "{3}{R/G}"
