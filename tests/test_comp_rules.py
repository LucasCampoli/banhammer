from banhammer.ingest.comp_rules import rules_to_documents

# Table of contents first, then the body, then the glossary: the parser keeps
# only what lies between the second "1. Game Concepts" and the second "Glossary".
RULES_TEXT = """\
Contents
1. Game Concepts
2. Parts of a Card
Glossary

1. Game Concepts

100. General

100.1. These Magic rules apply to any Magic game with two or more players.

100.1a A two-player game is a game that begins with only two players.

2. Parts of a Card

200. General

200.1. The parts of a card are name, mana cost, illustration, and type line.
Example: A card's name appears in its upper left corner.

Glossary

Abandon
To turn a face-up ongoing scheme card face down.
"""


def test_rules_to_documents_splits_rules_and_subrules():
    docs = {doc.id: doc for doc in rules_to_documents(RULES_TEXT)}
    assert list(docs) == ["100.1", "100.1a", "200.1"]
    assert docs["100.1a"].metadata["parent"] == "100.1"
    assert docs["100.1a"].metadata["section"] == "100. General"
    assert docs["200.1"].page_content.endswith("Example: A card's name appears in its upper left corner.")


def test_chapter_headings_are_not_glued_to_the_previous_rule():
    docs = {doc.id: doc for doc in rules_to_documents(RULES_TEXT)}
    assert docs["100.1a"].page_content == "100.1a A two-player game is a game that begins with only two players."
