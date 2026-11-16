import pytest

from banhammer.eval.cost import chat_usage, embedding_cost, embedding_tokens, price_for


def test_price_for_matches_dated_snapshots_by_longest_prefix():
    assert price_for("gpt-4o-mini-2024-07-18") == price_for("gpt-4o-mini")
    assert price_for("gpt-4o-2024-08-06") == price_for("gpt-4o")
    assert price_for("gpt-4o-mini") != price_for("gpt-4o")
    assert price_for("some-other-model") is None


def test_chat_usage_sums_models_and_discounts_cached_input():
    usage = {
        "gpt-4o-mini-2024-07-18": {
            "input_tokens": 1_000_000, "output_tokens": 100_000,
            "input_token_details": {"cache_read": 400_000},
        },
        "gpt-4o-mini": {"input_tokens": 0, "output_tokens": 100_000},
    }
    result = chat_usage(usage)
    assert result["input_tokens"] == 1_000_000
    assert result["cached_tokens"] == 400_000
    assert result["output_tokens"] == 200_000
    # 600k fresh * 0.15 + 400k cached * 0.075 + 200k out * 0.60, per million
    assert result["chat_cost_usd"] == pytest.approx(0.09 + 0.03 + 0.12)


def test_chat_usage_cost_is_unknown_for_an_unpriced_model():
    assert chat_usage({"mystery-model": {"input_tokens": 10, "output_tokens": 1}})["chat_cost_usd"] is None


def test_chat_usage_of_no_calls_is_free():
    assert chat_usage({}) == {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "chat_cost_usd": 0.0}


def test_embedding_tokens_and_cost():
    tokens = embedding_tokens(["sacrifice a creature", "draw a card"])
    assert tokens > 0
    assert embedding_cost(1_000_000, "text-embedding-3-small") == pytest.approx(0.02)
    assert embedding_cost(tokens, "unknown") is None


def test_get_llm_passes_only_the_params_the_config_sets(monkeypatch):
    from banhammer.providers.llm import get_llm

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    assert get_llm({"model": "gpt-4o-mini", "temperature": 0}).temperature == 0
    reasoning = get_llm({"model": "gpt-5.4-mini", "reasoning_effort": "low"})
    assert reasoning.temperature is None and reasoning.reasoning_effort == "low"


def test_retrieval_only_scores_combo_cards_without_a_judge(tmp_path, monkeypatch):
    import json

    from banhammer.eval import __main__ as eval_main
    from banhammer.models import Hit

    built = {}

    class Graph:
        def invoke(self, state):
            built["card"] = state["card"].name
            hit = Hit(id="1", namespace="cards", text="t", metadata={"name": "Pestermite"}, source="dense")
            return {"hits": [hit], "queries": []}

    def fake_build_graph(config, judge_report=True):
        built["judge_report"] = judge_report
        return Graph()

    monkeypatch.setattr(eval_main, "build_graph", fake_build_graph)
    config = tmp_path / "cfg.yaml"
    config.write_text("embeddings:\n  model: text-embedding-3-small\n", encoding="utf-8")
    gold = tmp_path / "gold.jsonl"
    rows = [{"card": {"name": "Twin"}, "combo_pieces": ["Pestermite", "Deceiver Exarch"], "expected_risk": "watch"},
            {"card": {"name": "Bolt"}, "combo_pieces": [], "expected_risk": "ok"}]
    gold.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    summary = eval_main.score_config(config, gold, tmp_path, retrieval_only=True)
    assert built["judge_report"] is False
    assert summary["cards"] == 1 and summary["recall@5"] == 0.5 and summary["risk_accuracy"] is None
    assert (tmp_path / "cfg-retrieval.json").exists()
    assert built["card"] == "Test Card"
    assert summary["groups"]["modern watch"]["recall@10"] == 0.5
