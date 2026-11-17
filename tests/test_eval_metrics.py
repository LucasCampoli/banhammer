from banhammer.eval.anonymize import ANONYMOUS_NAME, anonymize
from banhammer.eval.metrics import (balanced_accuracy, ban_detected, false_ban, ordinal_error, pieces_named,
                                     recall_at_k, wilson_interval)
from banhammer.models import BalanceReport, CardInput, Hit
from banhammer.pipeline.graph import exclude_self


def _hits(*names):
    return [Hit(id=name, namespace="cards", text=name, metadata={"name": name}, source="dense") for name in names]


def test_recall_counts_a_role_once_when_any_of_its_cards_is_found():
    hits = _hits("Sparring Mummy", "Opt")
    assert recall_at_k(hits, [["Pestermite", "Sparring Mummy"]], 10) == 1.0
    assert recall_at_k(hits, [["Pestermite", "Deceiver Exarch"], "Opt"], 10) == 0.5
    assert recall_at_k(hits, [["Pestermite", "Opt"]], 1) == 0.0


def test_ban_metrics_only_score_their_group():
    banned = BalanceReport(risk="likely_ban", confidence=1)
    watched = BalanceReport(risk="watch", confidence=1)
    assert ban_detected(banned, "likely_ban") is True and ban_detected(watched, "likely_ban") is False
    assert ban_detected(banned, "ok") is None
    assert false_ban(banned, "ok") is True and false_ban(banned, "watch") is True
    assert false_ban(watched, "ok") is False and false_ban(banned, "likely_ban") is None


def test_ordinal_error_counts_steps_between_levels():
    assert ordinal_error(BalanceReport(risk="likely_ban", confidence=1), "ok") == 2
    assert ordinal_error(BalanceReport(risk="watch", confidence=1), "likely_ban") == 1


def test_balanced_accuracy_scores_a_constant_answer_at_one_half():
    assert balanced_accuracy(1.0, 1.0) == 0.5 and balanced_accuracy(0.0, 0.0) == 0.5
    assert balanced_accuracy(0.8, 0.1) == 0.85
    assert balanced_accuracy(None, 0.1) is None


def test_pieces_named_reads_the_report_combos():
    report = BalanceReport(risk="watch", confidence=1, combos=[
        {"description": "Loops with Sanguine Bond.", "sequence": ["Cast Pestermite."]}])
    assert pieces_named(report, ["Sanguine Bond", ["Deceiver Exarch", "Pestermite"], "Triskelion"]) == 2 / 3


def test_wilson_interval_is_wide_for_a_small_gold_set():
    low, high = wilson_interval(18, 37)
    assert 0.33 < low < 0.34 and 0.64 < high < 0.65
    assert wilson_interval(0, 0) is None
    assert wilson_interval(0, 5)[0] == 0.0


def test_anonymize_hides_the_name_in_text_and_faces():
    card = CardInput(name="Kiki-Jiki, Mirror Breaker", oracle_text="Haste\nKiki-Jiki can't block.", oracle_id="oid")
    hidden = anonymize(card)
    assert hidden.name == ANONYMOUS_NAME and hidden.oracle_id == "oid"
    assert hidden.oracle_text == f"Haste\n{ANONYMOUS_NAME} can't block."
    split = CardInput(name="Fire // Ice", card_faces=[{"name": "Fire", "oracle_text": "Fire deals 2 damage."},
                                                      {"name": "Ice", "oracle_text": "Tap target permanent."}])
    faces = anonymize(split).card_faces
    assert [f.name for f in faces] == [f"{ANONYMOUS_NAME} (1)", f"{ANONYMOUS_NAME} (2)"]
    assert faces[0].oracle_text == f"{ANONYMOUS_NAME} deals 2 damage."


def test_exclude_self_drops_an_anonymized_card_by_oracle_id():
    card = anonymize(CardInput(name="Splinter Twin", oracle_id="twin-id"))
    hits = [Hit(id="twin-id", namespace="cards", text="t", metadata={"name": "Splinter Twin"}, source="dense"),
            *_hits("Pestermite")]
    assert [hit.id for hit in exclude_self(card, hits)] == ["Pestermite"]
