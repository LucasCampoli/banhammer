"""LangGraph for one card: build queries, search, drop the card itself, LLM report.

Catalog searches are restricted to the audited format before retrieval (`format_filter`), and
every claim in the report must cite a retrieved hit; claims that do not are discarded. With
`decompose: true` the card is split into primitives first and the queries are expanded; the
results of several queries in one namespace are merged with reciprocal rank fusion. With
`hybrid: true` every catalog query also runs against a BM25 index of the catalog.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from banhammer.models import BalanceReport, CardInput, Hit, Query
from banhammer.pipeline.query import build_queries
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
- risk: ok, watch, or likely_ban, and a confidence between 0 and 1.

Cite the bracket labels exactly (C1, R2). Every claim must cite the hits it relies on.
Claims with no valid citation are discarded. Do not invent rules or cards that are not in the hits.
Cards marked (banned in ...) are banned in that format."""


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
    """A real card being re-checked should not come back as its own result."""
    names = card.names()
    kept = []
    for hit in hits:
        name = str(hit.metadata.get("name", "")).casefold()
        if name and name in names:
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


def _numbered(hits: list[Hit]) -> tuple[dict[str, str], str]:
    labels: dict[str, str] = {}
    lines: list[str] = []
    cards = rules = 0
    for hit in hits:
        if hit.namespace == "rules":
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


def judge(llm: Any, card: CardInput, hits: list[Hit], fmt: str = "modern") -> tuple[BalanceReport, int]:
    """Ask the LLM for a BalanceReport, then keep only the claims that cite real hits."""
    labels, numbered = _numbered(hits)
    prompt = f"Card:\n{describe(card)}\n\nHits:\n{numbered}"
    result = llm.with_structured_output(BalanceReport).invoke([
        SystemMessage(content=_SYSTEM.format(format=fmt.title())),
        HumanMessage(content=prompt),
    ])
    report = result if isinstance(result, BalanceReport) else BalanceReport.model_validate(result)
    return validate_citations(report, labels)


def _processed_dir(cfg: dict) -> Path:
    path = Path(cfg.get("processed_dir", "data/processed"))
    return path if path.is_absolute() else ROOT / path


def build_graph(config: dict | Path | str):
    load_env()
    cfg = config if isinstance(config, dict) else load_yaml(config)
    embeddings = get_embeddings(cfg["embeddings"])
    model = cfg["embeddings"].get("model") or "text-embedding-3-small"
    stores = StoreFactory(cfg["vector_store"], embeddings, model=model, index_cfg=cfg.get("index"))
    llm = get_llm(cfg["llm"])
    k = int(cfg.get("k", 10))
    bm25 = BM25Index(_processed_dir(cfg) / "cards.jsonl") if cfg.get("hybrid", False) else None

    def make_queries(state: PipelineState) -> dict:
        queries, analysis = build_queries(state["card"], state.get("format") or "modern", cfg, llm)
        return {"queries": queries, "analysis": analysis}

    def retrieve(state: PipelineState) -> dict:
        lists: dict[str, list[list[Hit]]] = {}
        for query in state.get("queries") or []:
            lists.setdefault(query.namespace, []).append(search(stores, query, k))
            if bm25 is not None and query.namespace == "cards":
                lists["cards"].append(bm25.search(query, k))
        return {"hits": fuse_by_namespace(lists)}

    def drop_self(state: PipelineState) -> dict:
        return {"hits": exclude_self(state["card"], state.get("hits") or [])}

    def generate(state: PipelineState) -> dict:
        report, dropped = judge(llm, state["card"], state.get("hits") or [], state.get("format") or "modern")
        return {"report": report, "dropped_claims": dropped}

    graph = StateGraph(PipelineState)
    graph.add_node("build_queries", make_queries)
    graph.add_node("retrieve", retrieve)
    graph.add_node("exclude_self", drop_self)
    graph.add_node("generate", generate)
    graph.add_edge(START, "build_queries")
    graph.add_edge("build_queries", "retrieve")
    graph.add_edge("retrieve", "exclude_self")
    graph.add_edge("exclude_self", "generate")
    graph.add_edge("generate", END)
    return graph.compile()
