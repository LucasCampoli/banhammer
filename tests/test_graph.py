from banhammer.models import BalanceReport, CardInput, Hit
from banhammer.pipeline.graph import judge, number_hits, remap_citations


def card_hit(hit_id: str, name: str, modern: str) -> Hit:
    return Hit(
        id=hit_id,
        namespace="cards",
        text=f"{name} | Creature | {{1}}{{G}} | ...",
        metadata={"name": name, "modern": modern},
    )


CARD = CardInput(name="Test Card", mana_cost="{1}{B}", type_line="Creature", oracle_text="...")


class FakeLLM:
    """Stands in for the chat model: returns a fixed report."""

    def __init__(self, report: BalanceReport) -> None:
        self.report = report

    def with_structured_output(self, _schema):
        return self

    def invoke(self, _messages):
        return self.report


def test_number_hits_marks_banned_cards():
    hits = [card_hit("a", "Hogaak, Arisen Necropolis", "banned"), card_hit("b", "Bloodghast", "legal")]
    _, block = number_hits(hits)
    lines = block.splitlines()
    assert lines[0].startswith("[C1] (banned in Modern) Hogaak")
    assert lines[1].startswith("[C2] Bloodghast")


def test_remap_citations_keeps_known_labels_and_drops_unknown_ones():
    labels = {"C1": "oracle-a", "R1": "702.19"}
    assert remap_citations(["C1", "[r1]", "C12", "Kitchen Finks"], labels) == ["oracle-a", "702.19"]


def test_judge_marks_a_combo_citing_only_unknown_labels_as_speculation():
    answer = BalanceReport.model_validate({
        "combos": [
            {"description": "real", "card_ids": ["C1"]},
            {"description": "invented", "card_ids": ["C12"]},
        ],
        "risk": "watch",
        "confidence": 0.5,
    })
    report = judge(FakeLLM(answer), card=CARD, hits=[card_hit("oracle-a", "Bloodghast", "legal")])
    real, invented = report.combos
    assert real.card_ids == ["oracle-a"] and not real.speculation
    assert invented.card_ids == [] and invented.speculation
