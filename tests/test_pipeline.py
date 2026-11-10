from banhammer.eval.metrics import false_alarm, mean, recall_at_k
from banhammer.ingest.scryfall_cards import card_to_document
from banhammer.models import BalanceReport, CardInput, Combo, Hit, PowerAssessment, WordingIssue
from banhammer.pipeline.graph import judge
from banhammer.pipeline.retrieval import format_filter
from banhammer.shared.cardtext import type_words
from fakes import FakeLLM


def _hit(hit_id, namespace, name="", banned=None):
    metadata = {"name": name} if name else {}
    if banned:
        metadata["banned"] = banned
    return Hit(id=hit_id, namespace=namespace, text=f"{hit_id} text", metadata=metadata, source="dense")


def test_judge_drops_uncited_and_invented_claims():
    raw = BalanceReport(
        wording_issues=[
            WordingIssue(description="grounded", rule_ids=["R1"]),
            WordingIssue(description="no citation"),
            WordingIssue(description="invented rule", rule_ids=["R9"]),
            WordingIssue(description="card cited as a rule", rule_ids=["C1"]),
        ],
        combos=[
            Combo(description="grounded", card_ids=["[C2]", "C7"]),
            Combo(description="no citation"),
        ],
        power_assessment=PowerAssessment(summary="fine", comparable_ids=["C1", "C42"]),
        risk="watch",
        confidence=0.5,
    )
    hits = [_hit("oid-1", "cards", "A"), _hit("oid-2", "cards", "B"), _hit("702.19", "rules")]
    report, dropped = judge(FakeLLM(raw), CardInput(name="X"), hits, "standard")
    assert [i.description for i in report.wording_issues] == ["grounded"]
    assert report.wording_issues[0].rule_ids == ["702.19"]
    assert [c.card_ids for c in report.combos] == [["oid-2"]]
    assert report.power_assessment.comparable_ids == ["oid-1"]
    assert dropped == 4


def test_judge_prompt_names_format_and_flags_banned_cards():
    llm = FakeLLM(BalanceReport(risk="ok", confidence=1))
    judge(llm, CardInput(name="X"), [_hit("oid-1", "cards", "Pod", banned=["modern"])], "standard")
    system, human = (m.content for m in llm.messages)
    assert "Standard format" in system
    assert "[C1] (banned in modern) oid-1 text" in human


def test_format_filter():
    assert format_filter("standard") == {"formats": {"$in": ["standard"]}}


def _scryfall(**legalities):
    return {"oracle_id": "oid", "name": "Bear", "type_line": "Legendary Creature — Bear",
            "mana_cost": "{1}{G}", "oracle_text": "", "cmc": 2, "legalities": legalities}


def test_ingest_keeps_formats_and_bans():
    doc = card_to_document(_scryfall(modern="banned", standard="legal", legacy="legal"))
    assert doc.metadata["formats"] == ["modern", "standard"]
    assert doc.metadata["banned"] == ["modern"]
    assert doc.metadata["type_words"] == ["Legendary", "Creature"]
    assert card_to_document(_scryfall(modern="not_legal", standard="not_legal")) is None


def test_type_words_joins_faces():
    assert type_words("Creature — Human // Land") == ["Creature", "Land"]


def test_metrics():
    hits = [_hit("1", "rules"), _hit("2", "cards", "Kitchen Finks"), _hit("3", "cards", "Viscera Seer")]
    assert recall_at_k(hits, ["Kitchen Finks", "Murderous Redcap"], k=1) == 0.5
    assert recall_at_k(hits, ["kitchen finks", "viscera seer"], k=10) == 1.0
    flagged = BalanceReport(risk="watch", confidence=0.5)
    assert false_alarm(flagged, "ok") is True
    assert false_alarm(flagged, "likely_ban") is None
    assert mean([True, False, None]) == 0.5
    assert mean([None]) is None
