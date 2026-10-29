"""LangGraph for one card: full text, dense search, drop the card itself, LLM report."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from banhammer.models import BalanceReport, CardInput, Hit, Query
from banhammer.pipeline.state import PipelineState
from banhammer.providers import StoreFactory, get_embeddings, get_llm
from banhammer.shared.config import load_env, load_yaml

_SYSTEM = """You are a Magic: The Gathering judge for the Modern format.
You are given a card and numbered reference hits. Card hits are labeled [C1], [C2], ...
and Comprehensive Rules hits are labeled [R1], [R2], ...

Write a balance report:
- wording_issues: problems with how the card is written. Cite the rule labels you rely on in rule_ids.
- combos: dangerous interactions. Cite the card labels involved in card_ids and list the sequence of plays.
- power_assessment: power versus cost compared with the comparable cards retrieved. Cite those cards in comparable_ids.
- risk: ok, watch, or likely_ban, and a confidence between 0 and 1.

Cite the bracket labels exactly (C1, R2). Every claim must cite the hits it relies on.
A claim with no citation is speculation. Do not invent rules or cards that are not in the hits."""


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
        labels[label.casefold()] = hit.id
        lines.append(f"[{label}] {hit.text}")
    return labels, "\n".join(lines) if lines else "(no hits)"


def _remap(citations: list[str], labels: dict[str, str]) -> list[str]:
    remapped = []
    for citation in citations:
        token = citation.strip().strip("[]")
        remapped.append(labels.get(token) or labels.get(token.casefold()) or token)
    return remapped


def judge(llm: Any, card: CardInput, hits: list[Hit]) -> BalanceReport:
    """Ask the LLM for a BalanceReport. Bracket labels become hit ids."""
    labels, numbered = _numbered(hits)
    prompt = f"Card:\n{describe(card)}\n\nHits:\n{numbered}"
    result = llm.with_structured_output(BalanceReport).invoke([
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=prompt),
    ])
    report = result if isinstance(result, BalanceReport) else BalanceReport.model_validate(result)
    data = report.model_dump()
    for issue in data["wording_issues"]:
        issue["rule_ids"] = _remap(issue["rule_ids"], labels)
    for combo in data["combos"]:
        combo["card_ids"] = _remap(combo["card_ids"], labels)
    data["power_assessment"]["comparable_ids"] = _remap(
        data["power_assessment"]["comparable_ids"], labels
    )
    return BalanceReport.model_validate(data)


def build_graph(config: dict | Path | str):
    load_env()
    cfg = config if isinstance(config, dict) else load_yaml(config)
    embeddings = get_embeddings(cfg["embeddings"])
    model = cfg["embeddings"].get("model") or "text-embedding-3-small"
    stores = StoreFactory(cfg["vector_store"], embeddings, model=model, index_cfg=cfg.get("index"))
    llm = get_llm(cfg["llm"])
    k = int(cfg.get("k", 10))

    def build_queries(state: PipelineState) -> dict:
        text = state["card"].as_text()
        return {"queries": [
            Query(text=text, namespace="cards"),
            Query(text=text, namespace="rules"),
        ]}

    def retrieve(state: PipelineState) -> dict:
        hits = []
        for query in state.get("queries") or []:
            hits.extend(search(stores, query, k))
        return {"hits": hits}

    def drop_self(state: PipelineState) -> dict:
        return {"hits": exclude_self(state["card"], state.get("hits") or [])}

    def generate(state: PipelineState) -> dict:
        return {"report": judge(llm, state["card"], state.get("hits") or [])}

    graph = StateGraph(PipelineState)
    graph.add_node("build_queries", build_queries)
    graph.add_node("retrieve", retrieve)
    graph.add_node("exclude_self", drop_self)
    graph.add_node("generate", generate)
    graph.add_edge(START, "build_queries")
    graph.add_edge("build_queries", "retrieve")
    graph.add_edge("retrieve", "exclude_self")
    graph.add_edge("exclude_self", "generate")
    graph.add_edge("generate", END)
    return graph.compile()
