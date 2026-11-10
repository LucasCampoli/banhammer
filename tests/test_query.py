from banhammer.models import Ability, CardAnalysis, CardInput, Counterpart
from banhammer.pipeline.query import ability_text, build_queries, decompose, expanded_queries
from fakes import FakeLLM

CARD = CardInput(name="Spite Engine", type_line="Enchantment", mana_cost="{1}{R}",
                 oracle_text="Whenever an opponent draws a card, create a Treasure token and deal 1 damage to them.")

ANALYSIS = CardAnalysis(
    abilities=[Ability(text=CARD.oracle_text, trigger="whenever an opponent draws a card",
                       effect="create a Treasure token and deal 1 damage to that player")],
    counterparts=[Counterpart(description="Each opponent draws a card.", card_type="Instant", max_mana_value=2)],
    rules_topics=["Treasure tokens"],
)


def test_ability_text_skips_empty_primitives():
    assert ability_text(ANALYSIS.abilities[0]) == (
        f"{CARD.oracle_text} | trigger: whenever an opponent draws a card"
        " | effect: create a Treasure token and deal 1 damage to that player"
    )


def test_expanded_queries_add_rules_and_counterpart_queries():
    queries = expanded_queries(CARD, ANALYSIS, "standard")
    assert [(q.namespace, q.text) for q in queries[:2]] == [("cards", CARD.as_text()), ("rules", CARD.as_text())]
    rules = [q.text for q in queries[2:] if q.namespace == "rules"]
    assert rules == [ability_text(ANALYSIS.abilities[0]), "Treasure tokens"]
    counterpart = queries[-1]
    assert counterpart.namespace == "cards"
    assert counterpart.text == "Each opponent draws a card."
    assert counterpart.filter == {"formats": {"$in": ["standard"]}}


def test_expanded_queries_cap_counterparts_and_topics():
    many = CardAnalysis(counterparts=[Counterpart(description=f"c{i}") for i in range(8)],
                        rules_topics=[f"t{i}" for i in range(8)])
    queries = expanded_queries(CARD, many, "modern", use_format_filter=False)
    assert sum(q.namespace == "cards" for q in queries) == 1 + 5
    assert sum(q.namespace == "rules" for q in queries) == 1 + 5
    assert all(q.filter is None for q in queries)


def test_decompose_uses_structured_output():
    llm = FakeLLM(ANALYSIS.model_dump())
    assert decompose(llm, CARD) == ANALYSIS
    assert llm.schema is CardAnalysis
    system, human = (m.content for m in llm.messages)
    assert "OTHER half of a combo" in system
    assert CARD.oracle_text in human


def test_build_queries_without_decompose_matches_version_1():
    queries, analysis = build_queries(CARD, "modern", {"format_filter": True}, llm=None)
    assert analysis is None
    assert [(q.namespace, q.filter) for q in queries] == [("cards", {"formats": {"$in": ["modern"]}}),
                                                          ("rules", None)]


def test_build_queries_with_decompose_keeps_analysis():
    queries, analysis = build_queries(CARD, "modern", {"decompose": True}, FakeLLM(ANALYSIS))
    assert analysis == ANALYSIS
    assert len(queries) == 2 + 2 + 1
