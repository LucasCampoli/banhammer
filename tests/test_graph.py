from banhammer.models import BalanceReport, CardInput, Hit, RiskFeatures
from banhammer.pipeline.graph import decide_risk, free_judge, judge, judge_context, number_hits, validate_hybrid, validate_names


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


def test_free_judge_keeps_combos_that_name_cards_from_memory():
    answer = BalanceReport.model_validate({
        "combos": [{"description": "infinite", "card_ids": ["Pestermite"], "sequence": []}],
        "risk": "likely_ban", "confidence": 0.8,
    })
    report, dropped = free_judge(FakeLLM(answer), card=CARD)
    assert dropped == 0
    assert report.combos[0].card_ids == ["Pestermite"]


def features(**overrides) -> RiskFeatures:
    base = {"combo_outcome": "none", "combo_cards_needed": 0, "power_vs_cost": "on_curve", "real_drawback": False}
    return RiskFeatures.model_validate({**base, **overrides})


def test_decide_risk_bans_a_fast_two_card_combo():
    fast = features(combo_outcome="infinite_resources", combo_cards_needed=2, earliest_win_turn=3)
    assert decide_risk(fast)[0] == "likely_ban"


def test_decide_risk_watches_a_slow_combo_and_an_incremental_one_is_ok():
    slow = features(combo_outcome="wins_game", combo_cards_needed=2, earliest_win_turn=4)
    value = features(combo_outcome="incremental_value", combo_cards_needed=2, earliest_win_turn=3)
    assert decide_risk(slow)[0] == "watch"
    assert decide_risk(value)[0] == "ok"


def test_decide_risk_bans_raw_power_only_without_a_drawback():
    assert decide_risk(features(power_vs_cost="far_above_curve"))[0] == "likely_ban"
    assert decide_risk(features(power_vs_cost="far_above_curve", real_drawback=True))[0] == "watch"


def test_validate_names_drops_combos_naming_only_unknown_cards():
    report = BalanceReport.model_validate({
        "combos": [{"description": "made up", "card_ids": ["Pillow Fort"]},
                   {"description": "real", "card_ids": ["Pestermite", "Pillow Fort"]},
                   {"description": "alone", "card_ids": []}],
        "power_assessment": {"summary": "", "comparable_ids": ["Twinflame", "Not A Card"]},
        "risk": "watch", "confidence": 0.5,
    })
    cleaned, dropped = validate_names(report, {"pestermite", "twinflame"})
    assert dropped == 1
    assert [c.card_ids for c in cleaned.combos] == [["Pestermite"], []]
    assert cleaned.power_assessment.comparable_ids == ["Twinflame"]


def test_validate_names_ignores_the_card_naming_itself():
    report = BalanceReport.model_validate({
        "combos": [{"description": "alone", "card_ids": ["Test Card"]}],
        "risk": "watch", "confidence": 0.5,
    })
    cleaned, dropped = validate_names(report, set(), frozenset({"test card"}))
    assert dropped == 0 and cleaned.combos[0].card_ids == []


def test_validate_hybrid_accepts_labels_legal_names_and_rule_numbers():
    report = BalanceReport.model_validate({
        "wording_issues": [{"description": "rule", "rule_ids": ["702.19a"]},
                           {"description": "none", "rule_ids": ["made up"]}],
        "combos": [{"description": "hit", "card_ids": ["C1"]},
                   {"description": "memory", "card_ids": ["Pestermite"]},
                   {"description": "invented", "card_ids": ["Pillow Fort", "Test Card"]}],
        "risk": "watch", "confidence": 0.5,
    })
    cleaned, dropped = validate_hybrid(report, {"C1": "oracle-a"}, {"pestermite"}, frozenset({"test card"}))
    assert dropped == 2
    assert [c.card_ids for c in cleaned.combos] == [["oracle-a"], ["Pestermite"]]
    assert cleaned.wording_issues[0].rule_ids == ["702.19a"]
