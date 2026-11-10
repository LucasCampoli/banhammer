from banhammer.models import Hit
from banhammer.pipeline.graph import _numbered


def card_hit(hit_id: str, name: str, modern: str) -> Hit:
    return Hit(
        id=hit_id,
        namespace="cards",
        text=f"{name} | Creature | {{1}}{{G}} | ...",
        metadata={"name": name, "modern": modern},
        source="dense",
    )


def test_numbered_marks_banned_cards():
    hits = [card_hit("a", "Hogaak, Arisen Necropolis", "banned"), card_hit("b", "Bloodghast", "legal")]
    _, block = _numbered(hits)
    lines = block.splitlines()
    assert lines[0].startswith("[C1] (banned in Modern) Hogaak")
    assert lines[1].startswith("[C2] Bloodghast")
