import pytest

from banhammer.ingest.comp_rules import rules_to_documents

# Table of contents first, then the body. The parser keeps only what lies
# between the second "1. Game Concepts" and the second "8. Multiplayer Rules".
RULES_TEXT = """\
Contents
1. Game Concepts
2. Parts of a Card
8. Multiplayer Rules
9. Casual Variants
Glossary

1. Game Concepts

100. General

100.1. These Magic rules apply to any Magic game with two or more players.

100.1a A two-player game is a game that begins with only two players.

2. Parts of a Card

200. General

200.1. The parts of a card are name, mana cost, illustration, and type line.
Example: A card's name appears in its upper left corner.

8. Multiplayer Rules

800. General

800.1. The multiplayer rules are an optional set of additional rules.

9. Casual Variants

903. Commander

903.1. In the Commander variant, each deck is led by a legendary creature.

Glossary

Abandon
To turn a face-up ongoing scheme card face down.
"""


def test_rules_to_documents_splits_rules_and_subrules():
    docs = {doc.id: doc for doc in rules_to_documents(RULES_TEXT)}
    assert docs["100.1a"].metadata["parent"] == "100.1"
    assert docs["100.1a"].metadata["section"] == "100. General"
    assert docs["200.1"].page_content.endswith("Example: A card's name appears in its upper left corner.")


def test_multiplayer_and_casual_variant_rules_are_skipped():
    docs = rules_to_documents(RULES_TEXT)
    assert [doc.id for doc in docs] == ["100.1", "100.1a", "200.1"]


def test_chapter_headings_are_not_glued_to_the_previous_rule():
    docs = {doc.id: doc for doc in rules_to_documents(RULES_TEXT)}
    assert docs["100.1a"].page_content == "100.1a A two-player game is a game that begins with only two players."


def test_missing_multiplayer_chapter_is_an_error():
    without_chapter_8 = RULES_TEXT.replace("8. Multiplayer Rules", "8. Something Else")
    with pytest.raises(ValueError):
        rules_to_documents(without_chapter_8)
