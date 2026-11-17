from banhammer.models import BalanceReport, CardInput, Hit
from banhammer.pipeline.graph import judge, judge_context, number_hits


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


def test_number_hits_titles_similar_cards_and_combo_candidates_apart():
    combo = card_hit("p", "Pestermite").model_copy(update={"purpose": "combo"})
    rule = Hit(id="r", namespace="rules", text="702.1 ...", source="dense")
    labels, block = number_hits([combo, card_hit("s", "Twinflame"), rule])
    lines = block.splitlines()
    assert lines[0].startswith("Comparable cards") and lines[1].startswith("[C1] Twinflame")
    assert lines[2].startswith("Combo candidates") and lines[3].startswith("[C2] Pestermite")
    assert lines[4] == "Rules:" and labels == {"C1": "s", "C2": "p", "R1": "r"}


def test_judge_context_keeps_the_first_cards_of_each_purpose():
    combos = [card_hit(f"c{i}", f"Combo {i}").model_copy(update={"purpose": "combo"}) for i in range(3)]
    similar = [card_hit(f"s{i}", f"Similar {i}") for i in range(3)]
    kept = judge_context(combos + similar, 2)
    assert [hit.id for hit in kept] == ["c0", "c1", "s0", "s1"]


def test_judge_context_caps_the_rules():
    rules = [Hit(id=f"r{i}", namespace="rules", text="...", source="dense") for i in range(4)]
    assert [hit.id for hit in judge_context(rules + [card_hit("s", "Opt")], 5, 2)] == ["r0", "r1", "s"]
    assert len(judge_context(rules, None)) == 4
