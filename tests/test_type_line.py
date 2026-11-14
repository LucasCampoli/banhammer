from banhammer.ocr.type_line import TypeVocabulary, correct_type_line, correct_word

VOCABULARY = TypeVocabulary(
    head=frozenset({"Legendary", "Snow", "Basic", "Creature", "Land", "Instant", "Artifact"}),
    subtypes=frozenset({"Elf", "Warrior", "Vampire", "Island", "Swamp", "Beast", "Scorpion", "Imp"}),
)


def test_correct_word_needs_a_unique_close_match():
    assert correct_word("EIf", VOCABULARY.words) == "Elf"
    assert correct_word("Yampire", VOCABULARY.words) == "Vampire"
    assert correct_word("Hound", VOCABULARY.words) == "Hound"  # errata, not a misread
    assert correct_word("Imq", frozenset({"Imp", "Ima"})) == "Imq"  # two candidates at distance 1


def test_correct_type_line_fixes_words_and_dashes():
    assert correct_type_line("Legendary Creature — EIf Warrior", VOCABULARY) == "Legendary Creature — Elf Warrior"
    assert correct_type_line("Basic Snow Land — [Island", VOCABULARY) == "Basic Snow Land — Island"
    assert correct_type_line("Creature —Beast", VOCABULARY) == "Creature — Beast"
    assert correct_type_line("Creature - — Imp", VOCABULARY) == "Creature — Imp"


def test_correct_type_line_restores_a_missing_dash():
    assert correct_type_line("Creature Scorpion", VOCABULARY) == "Creature — Scorpion"
    assert correct_type_line("Instant", VOCABULARY) == "Instant"
