from banhammer.models import BalanceReport, CardInput, Hit
from banhammer.pipeline.graph import judge, number_hits


def card_hit(hit_id: str, name: str, banned: list[str] | None = None) -> Hit:
    return Hit(
        id=hit_id,
        namespace="cards",
        text=f"{name} | Creature | {{1}}{{G}} | ...",
        metadata={"name": name, "banned": banned or []},
        source="dense",
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
    hits = [card_hit("a", "Hogaak, Arisen Necropolis", ["modern"]), card_hit("b", "Bloodghast")]
    _, block = number_hits(hits)
    lines = block.splitlines()
    assert lines[0].startswith("[C1] (banned in modern) Hogaak")
    assert lines[1].startswith("[C2] Bloodghast")


def test_judge_drops_a_combo_citing_only_unknown_labels():
    answer = BalanceReport.model_validate({
        "combos": [
            {"description": "real", "card_ids": ["C1"]},
            {"description": "invented", "card_ids": ["C12"]},
        ],
        "risk": "watch",
        "confidence": 0.5,
    })
    report, dropped = judge(FakeLLM(answer), card=CARD, hits=[card_hit("oracle-a", "Bloodghast")])
    assert [combo.card_ids for combo in report.combos] == [["oracle-a"]]
    assert dropped == 1
