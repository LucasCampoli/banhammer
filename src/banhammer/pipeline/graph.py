"""LangGraph for one card: build queries, search, drop the card itself, LLM report.

Catalog searches are restricted to the audited format before retrieval (`card_filter`), and
every claim in the report must cite a retrieved hit; claims that do not are discarded. With
`decompose: true` the card is split into primitives first and the queries are expanded; the
results of several queries in one namespace are merged with reciprocal rank fusion. With
`hybrid: true` every catalog query also runs against a BM25 index of the catalog. With
`rerank: llm | cross_encoder` the first `candidates` cards are reordered, and the judge sees only
the first `context_cards` of them. `card_k` (default `k`) sets how many cards each catalog query
returns, so the reranker can see a wider pool than the rules queries. `decompose_llm` (default
`llm`) sets the model that writes the expanded queries; the reranker and the judge keep `llm`.
`retrieve: false` sends no queries, so the judge reads the card alone: the baseline without RAG.
With `split_context: true` the cards similar to the evaluated one and the combo candidates found by
the counterpart queries stay apart: they are fused separately, only the candidates are reranked,
and the judge sees the first `context_cards` of each in its own block. `context_rules` caps the rule
hits the judge sees, in fused order (all of them by default). `oracle_hits` in the input state go to the
judge as combo candidates, untouched by retrieval: the eval uses them to give it the known combo pieces.
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from banhammer.models import BalanceReport, CardInput, Hit, Query
from banhammer.pipeline.query import build_queries
from banhammer.pipeline.rerank import cross_encoder_rerank, llm_rerank, rerank_cards
from banhammer.pipeline.retrieval import BM25Index, fuse_by_namespace
from banhammer.pipeline.state import PipelineState
from banhammer.providers import StoreFactory, get_embeddings, get_llm
from banhammer.shared.config import ROOT, load_env, load_yaml

_SYSTEM = """You are a Magic: The Gathering judge for the {format} format.
You are given a card and numbered reference hits. Card hits are labeled [C1], [C2], ...
and Comprehensive Rules hits are labeled [R1], [R2], ...

Write a balance report:
- wording_issues: problems with how the card is written. Cite the rule labels you rely on in rule_ids.
- combos: dangerous interactions. Cite the card labels involved in card_ids and list the sequence of plays.
- power_assessment: power versus cost compared with the comparable cards retrieved. Cite those cards in comparable_ids.
- risk_rationale: one or two sentences that apply the risk criteria below to this card.
- risk: ok, watch, or likely_ban, and a confidence between 0 and 1.

Risk criteria, for the {format} format:
- likely_ban: the card is far stronger than anything at its cost among the comparable cards, or it
  enables a cheap, fast combo that wins the game or makes infinite resources, or it is a near copy
  of a card that is banned in {format}. Banned cards in this format are usually like that.
- watch: the card is a known piece of a combo that needs several specific cards, or it is at the
  top of the power curve for its cost but has a real drawback or a clear answer.
- ok: its power is in line with the comparable cards, and it has no dangerous combo you can cite.
  Most cards are ok. Being efficient, popular or having synergies is not enough for watch:
  removal, counterspells, card draw and mana creatures at a fair rate are ok.
Choose watch only when you can name the concrete reason it is not ok, and choose likely_ban
whenever the card meets a likely_ban criterion, even if you are unsure.

Some card hits are marked (banned in ...). They cannot be played in those formats, so never use
them as combo pieces there. Use them only as precedents: a card close to a banned card is a warning sign.

Cite the bracket labels exactly (C1, R2). Every claim must cite the hits it relies on.
Claims with no valid citation are discarded. Do not invent rules or cards that are not in the hits."""

_SPLIT_NOTE = """

The card hits come in two blocks. Comparable cards are the existing cards most similar to this
one: compare its power against them in power_assessment. Combo candidates were searched as
cards that might complete a combo with it: look for dangerous interactions among them in combos.
Most candidates will not combo; cite only interactions that really work."""

_BLOCK_TITLES = {
    "similar": "Comparable cards (compare power against these):",
    "combo": "Combo candidates (look for dangerous interactions with these):",
    "rules": "Rules:",
}


def search(stores: StoreFactory, query: Query, k: int) -> list[Hit]:
    """Top-k similar documents in the query's namespace."""
    store = stores.get(query.namespace)
    kwargs = {"filter": query.filter} if query.filter else {}
    pairs = store.similarity_search_with_score(query.text, k=k, **kwargs)
    hits = []
    for doc, score in pairs:
        metadata = dict(doc.metadata or {})
        metadata.setdefault("text", doc.page_content)
        hit_id = doc.id or metadata.get("oracle_id") or metadata.get("rule") or ""
        hits.append(Hit(
            id=str(hit_id),
            namespace=query.namespace,
            text=doc.page_content,
            metadata=metadata,
            score=float(score),
            source="dense",
        ))
    return hits


def exclude_self(card: CardInput, hits: list[Hit]) -> list[Hit]:
    """A real card being re-checked should not come back as its own result.

    It is matched by name, or by oracle id when the card has one (an anonymized card keeps its id).
    """
    names = card.names()
    kept = []
    for hit in hits:
        name = str(hit.metadata.get("name", "")).casefold()
        if (name and name in names) or (card.oracle_id and hit.id == card.oracle_id):
            continue
        kept.append(hit)
    return kept


def describe(card: CardInput) -> str:
    details: list[str] = []
    if card.power or card.toughness:
        details.append(f"power/toughness {card.power}/{card.toughness}")
    if card.loyalty:
        details.append(f"loyalty {card.loyalty}")
    if card.cmc is not None:
        details.append(f"mana value {card.cmc:g}")
    if card.colors:
        details.append("colors " + " ".join(card.colors))
    if card.keywords:
        details.append("keywords " + ", ".join(card.keywords))
    text = card.as_text()
    return text if not details else f"{text}\n" + "; ".join(details)


def number_hits(hits: list[Hit]) -> tuple[dict[str, str], str]:
    """Label hits C1, C2, ... and R1, R2, ... Returns label -> hit id, and the prompt block.

    Cards similar to the evaluated one come first, then combo candidates, then rules. When there
    are combo candidates every block gets a title, so the judge knows what each card is for.
    """
    blocks = {
        "similar": [hit for hit in hits if hit.namespace == "cards" and hit.purpose == "similar"],
        "combo": [hit for hit in hits if hit.namespace == "cards" and hit.purpose == "combo"],
        "rules": [hit for hit in hits if hit.namespace == "rules"],
    }
    titled = bool(blocks["combo"])
    labels: dict[str, str] = {}
    lines: list[str] = []
    cards = rules = 0
    for block, block_hits in blocks.items():
        if titled and block_hits:
            lines.append(_BLOCK_TITLES[block])
        for hit in block_hits:
            if block == "rules":
                rules += 1
                label = f"R{rules}"
            else:
                cards += 1
                label = f"C{cards}"
            labels[label] = hit.id
            banned = hit.metadata.get("banned") or []
            flag = f"(banned in {', '.join(banned)}) " if banned else ""
            lines.append(f"[{label}] {flag}{hit.text}")
    return labels, "\n".join(lines) if lines else "(no hits)"


def judge_context(hits: list[Hit], context_cards: int | None, context_rules: int | None = None) -> list[Hit]:
    """The first ``context_cards`` card hits of each purpose and the first ``context_rules`` rule hits, in order.

    None keeps every hit of that kind.
    """
    kept: list[Hit] = []
    seen: dict[str, int] = {}
    for hit in hits:
        group = hit.purpose if hit.namespace == "cards" else "rules"
        limit = context_cards if hit.namespace == "cards" else context_rules
        seen[group] = seen.get(group, 0) + 1
        if limit is None or seen[group] <= limit:
            kept.append(hit)
    return kept


def _valid(citations: list[str], labels: dict[str, str], prefix: str) -> list[str]:
    """Hit ids for the citations that point at a real hit of the right kind. The rest are dropped."""
    ids = []
    for citation in citations:
        token = citation.strip().strip("[]").upper()
        if token.startswith(prefix) and token in labels and labels[token] not in ids:
            ids.append(labels[token])
    return ids


def validate_citations(report: BalanceReport, labels: dict[str, str]) -> tuple[BalanceReport, int]:
    """Keep only grounded claims. A wording issue needs a rule hit, a combo needs a card hit.

    Returns the cleaned report and how many claims were discarded.
    """
    issues, combos = [], []
    for issue in report.wording_issues:
        rule_ids = _valid(issue.rule_ids, labels, "R")
        if rule_ids:
            issues.append(issue.model_copy(update={"rule_ids": rule_ids}))
    for combo in report.combos:
        card_ids = _valid(combo.card_ids, labels, "C")
        if card_ids:
            combos.append(combo.model_copy(update={"card_ids": card_ids}))
    power = report.power_assessment.model_copy(
        update={"comparable_ids": _valid(report.power_assessment.comparable_ids, labels, "C")}
    )
    dropped = len(report.wording_issues) - len(issues) + len(report.combos) - len(combos)
    cleaned = report.model_copy(update={"wording_issues": issues, "combos": combos, "power_assessment": power})
    return cleaned, dropped


def judge(llm: BaseChatModel, card: CardInput, hits: list[Hit], fmt: str = "modern") -> tuple[BalanceReport, int]:
    """Ask the LLM for a BalanceReport, then keep only the claims that cite real hits."""
    labels, numbered = number_hits(hits)
    prompt = f"Card:\n{describe(card)}\n\nHits:\n{numbered}"
    system = _SYSTEM.format(format=fmt.title())
    if any(hit.purpose == "combo" for hit in hits):
        system += _SPLIT_NOTE
    result = llm.with_structured_output(BalanceReport).invoke([
        SystemMessage(content=system),
        HumanMessage(content=prompt),
    ])
    report = result if isinstance(result, BalanceReport) else BalanceReport.model_validate(result)
    return validate_citations(report, labels)


RERANKERS = ("none", "llm", "cross_encoder")


def _processed_dir(cfg: dict) -> Path:
    path = Path(cfg.get("processed_dir", "data/processed"))
    return path if path.is_absolute() else ROOT / path


def build_graph(config: dict | Path | str, judge_report: bool = True):
    """The pipeline for one card. Without `judge_report` it stops after reranking (no report), to measure retrieval alone."""
    load_env()
    cfg = config if isinstance(config, dict) else load_yaml(config)
    embeddings = get_embeddings(cfg["embeddings"])
    model = cfg["embeddings"].get("model") or "text-embedding-3-small"
    stores = StoreFactory(embeddings, model=model)
    llm = get_llm(cfg["llm"])
    # The decomposition writes the counterpart queries and drives recall; it can use a stronger model.
    decompose_llm = get_llm(cfg["decompose_llm"]) if cfg.get("decompose_llm") else llm
    k = int(cfg.get("k", 10))
    # Catalog queries can go wider than rules queries: every rule hit reaches the judge, cards are cut later.
    card_k = int(cfg.get("card_k", k))
    bm25 = BM25Index(_processed_dir(cfg) / "cards.jsonl") if cfg.get("hybrid", False) else None
    reranker = str(cfg.get("rerank") or "none")
    if reranker not in RERANKERS:
        raise ValueError(f"rerank must be one of {', '.join(RERANKERS)}, got {reranker!r}")
    candidates = int(cfg.get("candidates", 25))
    context_cards = cfg.get("context_cards")
    context_cards = None if context_cards is None else int(context_cards)
    split_context = bool(cfg.get("split_context", False))
    context_rules = cfg.get("context_rules")
    context_rules = None if context_rules is None else int(context_rules)

    def make_queries(state: PipelineState) -> dict:
        if not cfg.get("retrieve", True):
            return {"queries": [], "analysis": None}
        queries, analysis = build_queries(state["card"], state.get("format") or "modern", cfg, decompose_llm)
        return {"queries": queries, "analysis": analysis}

    def retrieve(state: PipelineState) -> dict:
        lists: dict[str, list[list[Hit]]] = {}
        for query in state.get("queries") or []:
            query_k = card_k if query.namespace == "cards" else k
            found = search(stores, query, query_k)
            if bm25 is not None and query.namespace == "cards":
                found = [found, bm25.search(query, query_k)]
            else:
                found = [found]
            key = query.namespace
            if split_context and query.namespace == "cards":
                # Similar cards and combo candidates are fused apart, so neither crowds out the other.
                key = f"cards:{query.purpose}"
                found = [[hit.model_copy(update={"purpose": query.purpose}) for hit in hits] for hits in found]
            lists.setdefault(key, []).extend(found)
        return {"hits": fuse_by_namespace(lists)}

    def drop_self(state: PipelineState) -> dict:
        return {"hits": exclude_self(state["card"], state.get("hits") or [])}

    def rerank(state: PipelineState) -> dict:
        if reranker == "none":
            return {}
        card = state["card"]
        if reranker == "llm":
            def reorder(hits: list[Hit]) -> list[Hit]:
                return llm_rerank(llm, card, hits)
        else:
            def reorder(hits: list[Hit]) -> list[Hit]:
                return cross_encoder_rerank(card, hits)
        hits = state.get("hits") or []
        if split_context:
            # Only combo candidates are reranked; similar cards keep their similarity order.
            combo = [hit for hit in hits if hit.purpose == "combo"]
            return {"hits": rerank_cards(combo, reorder, candidates)
                    + [hit for hit in hits if hit.purpose != "combo"]}
        return {"hits": rerank_cards(hits, reorder, candidates)}

    def generate(state: PipelineState) -> dict:
        hits = (state.get("oracle_hits") or []) + judge_context(state.get("hits") or [], context_cards, context_rules)
        report, dropped = judge(llm, state["card"], hits, state.get("format") or "modern")
        return {"report": report, "dropped_claims": dropped}

    graph = StateGraph(PipelineState)
    graph.add_node("build_queries", make_queries)
    graph.add_node("retrieve", retrieve)
    graph.add_node("exclude_self", drop_self)
    graph.add_node("rerank", rerank)
    if judge_report:
        graph.add_node("generate", generate)
    graph.add_edge(START, "build_queries")
    graph.add_edge("build_queries", "retrieve")
    graph.add_edge("retrieve", "exclude_self")
    graph.add_edge("exclude_self", "rerank")
    if judge_report:
        graph.add_edge("rerank", "generate")
        graph.add_edge("generate", END)
    else:
        graph.add_edge("rerank", END)
    return graph.compile()


def evaluate_card(graph, card: CardInput, fmt: str = "modern") -> tuple[BalanceReport, dict]:
    """Run a compiled pipeline on one card; returns the report and the final state (hits, dropped claims)."""
    result = graph.invoke({"card": card, "format": fmt, "queries": [], "hits": []})
    report = result["report"]
    if not isinstance(report, BalanceReport):
        report = BalanceReport.model_validate(report)
    return report, result
