from banhammer.models import BalanceReport, Hit
from banhammer.shared.report import cited_sources


def test_cited_sources_keeps_only_cited_hits_with_their_text():
    report = BalanceReport.model_validate({
        "wording_issues": [{"description": "vague", "rule_ids": ["111.12"]}],
        "combos": [{"description": "loop", "card_ids": ["pestermite"]}],
        "power_assessment": {"summary": "strong", "comparable_ids": ["twinflame"]},
        "risk": "likely_ban",
        "confidence": 0.9,
    })
    hits = [
        Hit(id="pestermite", namespace="cards", text="Pestermite | ...", metadata={"name": "Pestermite"}, source="dense"),
        Hit(id="twinflame", namespace="cards", text="Twinflame | ...", metadata={"name": "Twinflame"}, source="dense"),
        Hit(id="111.12", namespace="rules", text="111.12 A copy of a token ...", source="dense"),
        Hit(id="storm-crow", namespace="cards", text="Storm Crow | ...", metadata={"name": "Storm Crow"}, source="dense"),
    ]
    sources = cited_sources(report, hits)
    assert set(sources) == {"pestermite", "twinflame", "111.12"}
    assert sources["pestermite"] == {"namespace": "cards", "name": "Pestermite", "text": "Pestermite | ..."}
    assert sources["111.12"]["name"] == "111.12"
