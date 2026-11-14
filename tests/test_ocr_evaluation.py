from banhammer.ocr.evaluation import cer, flatten, levenshtein, normalize_rules, normalize_type_line


def test_levenshtein_and_cer():
    assert levenshtein("kitten", "sitting") == 3
    assert cer("abc", "abc") == 0.0
    assert cer("", "") == 0.0
    assert cer("x", "") == 1.0
    assert cer("a much longer prediction", "ab") == 1.0


def test_flatten_unifies_whitespace_and_quotes():
    assert flatten("a\n b  ’c’") == "a b 'c'"
    assert flatten(None) == ""


def test_normalize_rules_matches_printed_and_oracle_templating():
    name = "Hogaak, Arisen Necropolis"
    printed = "When Hogaak enters the battlefield, draw a card."
    oracle = "When this creature enters, draw a card."
    assert normalize_rules(printed, name) == normalize_rules(oracle, name) == "When ~ enters, draw a card."


def test_normalize_rules_keeps_names_inside_other_words():
    assert normalize_rules("Pyromancer's Pyromancer", "Pyro") == "Pyromancer's Pyromancer"


def test_normalize_type_line_renames_tribal():
    assert normalize_type_line("Tribal Instant — Goblin") == "Kindred Instant — Goblin"
