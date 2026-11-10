from banhammer.models import CardInput, Hit
from banhammer.pipeline.graph import judge_context
from banhammer.pipeline.rerank import CandidateScore, CandidateScores, llm_rerank, rerank_cards
from fakes import FakeLLM

CARD = CardInput(name="X", oracle_text="Whenever an opponent draws a card, they lose 1 life.")


def _hit(hit_id, namespace="cards"):
    return Hit(id=hit_id, namespace=namespace, text=f"{hit_id} text", source="dense")


def _ids(hits):
    return [hit.id for hit in hits]


def _scores(**by_label):
    return CandidateScores(scores=[CandidateScore(label=label, score=score) for label, score in by_label.items()])


def test_llm_rerank_orders_by_score():
    llm = FakeLLM(_scores(C1=2, C2=9, C3=5))
    reranked = llm_rerank(llm, CARD, [_hit("a"), _hit("b"), _hit("c")])
    assert _ids(reranked) == ["b", "c", "a"]
    assert [hit.score for hit in reranked] == [9, 5, 2]
    assert "[C2] b text" in llm.messages[1].content


def test_llm_rerank_ties_keep_order_and_missing_labels_score_zero():
    llm = FakeLLM(_scores(C2=3, C3=3, C4=-1))
    reranked = llm_rerank(llm, CARD, [_hit("a"), _hit("b"), _hit("c"), _hit("d")])
    assert _ids(reranked) == ["b", "c", "a", "d"]


def test_llm_rerank_accepts_bracketed_labels_and_skips_empty_input():
    llm = FakeLLM({"scores": [{"label": "[c2]", "score": 7}]})
    assert _ids(llm_rerank(llm, CARD, [_hit("a"), _hit("b")])) == ["b", "a"]
    unused = FakeLLM(None)
    assert llm_rerank(unused, CARD, []) == []
    assert unused.messages is None


def test_rerank_cards_reorders_only_the_candidates():
    hits = [_hit("c1"), _hit("c2"), _hit("r1", "rules"), _hit("c3")]
    reranked = rerank_cards(hits, lambda cards: list(reversed(cards)), candidates=2)
    assert _ids(reranked) == ["c2", "c1", "c3", "r1"]


def test_judge_context_keeps_rules_and_top_cards():
    hits = [_hit("c1"), _hit("c2"), _hit("c3"), _hit("r1", "rules"), _hit("r2", "rules")]
    assert _ids(judge_context(hits, 2)) == ["c1", "c2", "r1", "r2"]
    assert judge_context(hits, None) == hits


class _Doc:
    def __init__(self, doc_id, name):
        self.id, self.page_content, self.metadata = doc_id, f"{name} text", {"name": name}


class _FakeStore:
    def __init__(self, namespace, calls):
        self.namespace, self.calls = namespace, calls

    def similarity_search_with_score(self, text, k, filter=None):
        self.calls.append((self.namespace, text, filter))
        prefix = "c" if self.namespace == "cards" else "r"
        return [(_Doc(f"{prefix}{i}", f"{prefix}{i}"), 1.0 - i / 10) for i in range(1, 4)]


def _run_graph(monkeypatch, cfg, llm):
    from banhammer.pipeline import graph as graph_module

    calls = []

    class Stores:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, namespace):
            return _FakeStore(namespace, calls)

    monkeypatch.setattr(graph_module, "load_env", lambda: None)
    monkeypatch.setattr(graph_module, "get_embeddings", lambda cfg: None)
    monkeypatch.setattr(graph_module, "get_llm", lambda cfg: llm)
    monkeypatch.setattr(graph_module, "StoreFactory", Stores)
    base = {"llm": {}, "embeddings": {}, "vector_store": {}, "k": 3}
    result = graph_module.build_graph({**base, **cfg}).invoke({"card": CARD, "format": "standard"})
    return calls, result


def test_version_1_graph_is_unchanged(monkeypatch):
    from banhammer.models import BalanceReport

    llm = FakeLLM(BalanceReport(risk="ok", confidence=1))
    calls, result = _run_graph(monkeypatch, {"format_filter": True}, llm)
    assert calls == [("cards", CARD.as_text(), {"formats": {"$in": ["standard"]}}),
                     ("rules", CARD.as_text(), None)]
    assert _ids(result["hits"]) == ["c1", "c2", "c3", "r1", "r2", "r3"]
    assert [hit.score for hit in result["hits"]] == [0.9, 0.8, 0.7, 0.9, 0.8, 0.7]
    assert result.get("analysis") is None


def test_llm_rerank_graph_sends_top_cards_to_judge(monkeypatch):
    from banhammer.models import BalanceReport

    class TwoStepLLM(FakeLLM):
        def with_structured_output(self, schema):
            self.schema = schema
            return self

        def invoke(self, messages):
            self.messages = messages
            if self.schema is CandidateScores:
                return _scores(C2=10)
            return BalanceReport(risk="ok", confidence=1)

    llm = TwoStepLLM(None)
    _, result = _run_graph(monkeypatch, {"rerank": "llm", "candidates": 2, "context_cards": 1}, llm)
    assert _ids(result["hits"]) == ["c2", "c1", "c3", "r1", "r2", "r3"]
    judge_prompt = llm.messages[1].content
    assert "[C1] c2 text" in judge_prompt and "c1 text" not in judge_prompt and "[R3] r3 text" in judge_prompt
