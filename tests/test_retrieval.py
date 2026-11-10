from banhammer.models import Hit
from banhammer.pipeline.retrieval import fuse_by_namespace, rrf


def _hit(hit_id, namespace="cards", source="dense", score=None):
    return Hit(id=hit_id, namespace=namespace, text=hit_id, source=source, score=score)


def _ids(hits):
    return [hit.id for hit in hits]


def test_rrf_single_list_keeps_order():
    assert _ids(rrf([[_hit("a"), _hit("b"), _hit("c")]])) == ["a", "b", "c"]


def test_rrf_rewards_agreement_and_deduplicates():
    fused = rrf([[_hit("a"), _hit("b"), _hit("c")], [_hit("b", source="bm25"), _hit("c", source="bm25")]])
    assert _ids(fused) == ["b", "c", "a"]
    by_id = {hit.id: hit for hit in fused}
    assert by_id["b"].score == 1 / 62 + 1 / 61
    assert by_id["b"].source == "bm25+dense"
    assert by_id["a"].source == "dense"


def test_fuse_by_namespace_leaves_single_lists_untouched():
    cards = [_hit("c1", score=0.9), _hit("c2", score=0.8)]
    rules = [[_hit("r1", "rules")], [_hit("r2", "rules"), _hit("r1", "rules")]]
    fused = fuse_by_namespace({"cards": [cards], "rules": rules})
    assert fused[:2] == cards
    assert _ids(fused[2:]) == ["r1", "r2"]
