import json

import pytest

from banhammer.models import Hit, Query
from banhammer.pipeline.retrieval import BM25Index, card_filter, fuse_by_namespace, matches, rrf, tokenize


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


def test_matches_operators():
    meta = {"formats": ["modern", "standard"], "type_words": ["Artifact", "Creature"], "cmc": 3.0}
    assert matches(meta, None)
    assert matches(meta, {"formats": {"$in": ["standard"]}})
    assert not matches(meta, {"formats": {"$in": ["legacy"]}})
    assert matches({"rarity": "rare"}, {"rarity": {"$in": ["rare", "mythic"]}})
    assert matches(meta, {"cmc": {"$lte": 3}})
    assert not matches(meta, {"cmc": {"$lte": 2.0}})
    assert not matches({}, {"cmc": {"$lte": 2.0}})
    assert matches(meta, {"$and": [{"formats": {"$in": ["modern"]}}, {"type_words": {"$in": ["Creature"]}}]})
    assert not matches(meta, {"$and": [{"formats": {"$in": ["modern"]}}, {"type_words": {"$in": ["Land"]}}]})
    with pytest.raises(ValueError):
        matches(meta, {"cmc": {"$gt": 1}})


def test_tokenize():
    assert tokenize("Sacrifice an artifact: Add {C}{C}.") == ["sacrifice", "an", "artifact", "add", "c", "c"]


def _card_row(oracle_id, name, text, formats):
    line = f"{name} | Creature | {{2}} | {text}"
    return {"id": oracle_id, "text": line, "metadata": {"name": name, "formats": formats, "cmc": 2.0}}


@pytest.fixture
def bm25(tmp_path):
    rows = [
        _card_row("1", "Bear", "Vanilla.", ["modern", "standard"]),
        _card_row("2", "Ghoul", "Sacrifice a creature: Scry 1.", ["modern"]),
        _card_row("3", "Seer", "Sacrifice a creature: Scry 1.", ["modern", "standard"]),
        _card_row("4", "Drake", "Flying.", ["standard"]),
    ]
    path = tmp_path / "cards.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return BM25Index(path)


def test_bm25_finds_keywords(bm25):
    hits = bm25.search(Query(text="sacrifice a creature", namespace="cards"), k=10)
    assert {hit.metadata["name"] for hit in hits[:2]} == {"Ghoul", "Seer"}
    assert all(hit.source == "bm25" and hit.namespace == "cards" for hit in hits)
    assert hits[0].metadata["text"] == hits[0].text


def test_bm25_respects_filter_and_k(bm25):
    query = Query(text="sacrifice a creature", namespace="cards", filter={"formats": {"$in": ["standard"]}})
    assert [hit.metadata["name"] for hit in bm25.search(query, k=1)] == ["Seer"]
    assert bm25.search(Query(text="!!!", namespace="cards"), k=5) == []


def test_card_filter_combinations():
    assert card_filter() is None
    assert card_filter("modern") == {"formats": {"$in": ["modern"]}}
    assert card_filter(card_type="creature") == {"type_words": {"$in": ["Creature"]}}
    assert card_filter(max_mana_value=3) == {"cmc": {"$lte": 3.0}}
    assert card_filter("standard", "artifact", 2) == {"$and": [
        {"formats": {"$in": ["standard"]}},
        {"type_words": {"$in": ["Artifact"]}},
        {"cmc": {"$lte": 2.0}},
    ]}


def test_matches_accepts_card_filter_output():
    meta = {"formats": ["modern"], "type_words": ["Artifact", "Creature"], "cmc": 2.0}
    assert matches(meta, card_filter("modern", "creature", 2))
    assert not matches(meta, card_filter("modern", "creature", 1))
    assert not matches(meta, card_filter("standard", "creature", 2))
    assert not matches(meta, card_filter("modern", "land"))
