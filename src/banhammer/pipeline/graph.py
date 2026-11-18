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
Adding `cite: false` swaps in a prompt with the same risk criteria but no citation rules, so the
judge may use what the model knows: the fair "LLM alone" baseline. `cite: catalog` does the same but
keeps only the cards it names that exist and are legal in the format. `decide: rule` (also without
retrieval) has the judge estimate RiskFeatures and sets the risk level with `decide_risk`.
`cite: hybrid` (with retrieval) lets the judge cite hits or name legal cards and rule numbers it knows.
With `split_context: true` the cards similar to the evaluated one and the combo candidates found by
the counterpart queries stay apart: they are fused separately, only the candidates are reranked,
and the judge sees the first `context_cards` of each in its own block. `context_rules` caps the rule
hits the judge sees, in fused order (all of them by default). `oracle_hits` in the input state go to the
judge as combo candidates, untouched by retrieval: the eval uses them to give it the known combo pieces.
"""
from __future__ import annotations

import re
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from banhammer.models import BalanceReport, CardInput, Combo, Hit, PowerAssessment, Query, RiskFeatures
from banhammer.pipeline.query import build_queries
from banhammer.pipeline.rerank import cross_encoder_rerank, llm_rerank, rerank_cards
from banhammer.pipeline.retrieval import BM25Index, fuse_by_namespace
from banhammer.pipeline.state import PipelineState
from banhammer.providers import StoreFactory, get_embeddings, get_llm
from banhammer.shared.config import ROOT, load_env, load_yaml
from banhammer.shared.documents import read_jsonl

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

# The judge without retrieval and without the citation rules: it may rely on what the model knows.
# The risk criteria are the same as in _SYSTEM, only rewritten to not refer to retrieved hits.
_FREE_SYSTEM = """You are a Magic: The Gathering judge for the {format} format.
You are given a card. There are no reference hits: rely on your own knowledge of existing Magic
cards and of the Comprehensive Rules.

Write a balance report:
- wording_issues: problems with how the card is written. Put the rule numbers you rely on in rule_ids.
- combos: dangerous interactions with existing cards legal in {format}. Put their names in card_ids
  and list the sequence of plays.
- power_assessment: power versus cost compared with existing cards at a similar cost. Put their names in comparable_ids.
- risk_rationale: one or two sentences that apply the risk criteria below to this card.
- risk: ok, watch, or likely_ban, and a confidence between 0 and 1.

Risk criteria, for the {format} format:
- likely_ban: the card is far stronger than any existing card at its cost, or it enables a cheap,
  fast combo that wins the game or makes infinite resources, or it is a near copy of a card that
  is banned in {format}. Banned cards in this format are usually like that.
- watch: the card is a known piece of a combo that needs several specific cards, or it is at the
  top of the power curve for its cost but has a real drawback or a clear answer.
- ok: its power is in line with comparable existing cards, and it has no dangerous combo you can name.
  Most cards are ok. Being efficient, popular or having synergies is not enough for watch:
  removal, counterspells, card draw and mana creatures at a fair rate are ok.
Choose watch only when you can name the concrete reason it is not ok, and choose likely_ban
whenever the card meets a likely_ban criterion, even if you are unsure.

Cards banned in {format} cannot be played there, so never use them as combo pieces. A card close
to a banned card is a warning sign."""

# The judge that estimates features instead of choosing the risk level. It does not see the rule.
_FEATURES_SYSTEM = """You are a Magic: The Gathering judge for the {format} format.
You are given a card. There are no reference hits: rely on your own knowledge of existing Magic
cards and of the Comprehensive Rules.

Estimate the facts asked for in the schema as accurately as you can. Be concrete and literal:
- Work out the combo step by step and check that it really loops: if each repetition costs mana
  or another resource, it is incremental_value, not infinite_resources.
- Name only existing cards that are legal in {format}. Cards banned there cannot be combo pieces.
- Most cards have no dangerous combo and are on curve."""

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


def legal_names(cards_path: Path, fmt: str) -> set[str]:
    """Casefolded names of the catalog cards legal (not banned) in `fmt`, faces of split cards included."""
    names: set[str] = set()
    for doc in read_jsonl(cards_path):
        meta = doc.metadata
        if fmt in (meta.get("formats") or []) and fmt not in (meta.get("banned") or []):
            full = str(meta.get("name", ""))
            names.add(full.casefold())
            names.update(part.strip().casefold() for part in full.split("//"))
    return names


def validate_names(report: BalanceReport, legal: set[str], own: frozenset[str] = frozenset()) -> tuple[BalanceReport, int]:
    """Keep only card names that exist and are legal in the format, for a judge that names cards from memory.

    `own` holds the evaluated card's names (casefolded): the judge may name the card itself, which
    is neither kept nor counted as unknown. A combo whose other cards are all unknown or illegal is
    discarded; a combo that names no other card (the card alone) is kept. Returns the cleaned report
    and how many combos were discarded.
    """
    def known(names: list[str]) -> list[str]:
        return [name for name in names if name.strip().casefold() in legal]

    combos = []
    for combo in report.combos:
        others = [name for name in combo.card_ids if name.strip().casefold() not in own]
        valid = known(others)
        if valid or not others:
            combos.append(combo.model_copy(update={"card_ids": valid}))
    power = report.power_assessment.model_copy(
        update={"comparable_ids": known(report.power_assessment.comparable_ids)})
    cleaned = report.model_copy(update={"combos": combos, "power_assessment": power})
    return cleaned, len(report.combos) - len(combos)


def decide_risk(features: RiskFeatures) -> tuple[str, str]:
    """The risk level from the judge's estimates, by a rule fixed before any run. Returns (risk, why).

    Modern's own policy is that no deck should win consistently before turn four, so a cheap
    two-card combo that wins or loops by turn three is likely_ban, and so is raw power far above
    the curve with no real drawback. Any game-winning or looping combo, or power above the curve,
    is watch. Everything else is ok.
    """
    loops = features.combo_outcome in ("wins_game", "infinite_resources")
    fast = features.earliest_win_turn is not None and features.earliest_win_turn <= 3
    if loops and 0 < features.combo_cards_needed <= 2 and fast:
        return "likely_ban", "a combo of two cards or fewer wins or loops by turn three"
    if features.power_vs_cost == "far_above_curve" and not features.real_drawback:
        return "likely_ban", "far above the curve with no real drawback"
    if loops:
        return "watch", "a combo wins or loops, but it is slower or needs more cards"
    if features.power_vs_cost in ("above_curve", "far_above_curve"):
        return "watch", "above the curve"
    return "ok", "no fast combo and power on curve"


def feature_judge(llm: BaseChatModel, card: CardInput, fmt: str = "modern") -> BalanceReport:
    """Ask the LLM for RiskFeatures from the card alone, and turn them into a report with decide_risk."""
    result = llm.with_structured_output(RiskFeatures).invoke([
        SystemMessage(content=_FEATURES_SYSTEM.format(format=fmt.title())),
        HumanMessage(content=f"Card:\n{describe(card)}"),
    ])
    features = result if isinstance(result, RiskFeatures) else RiskFeatures.model_validate(result)
    risk, why = decide_risk(features)
    combos = [] if features.combo_outcome == "none" or not features.best_combo else [
        Combo(description=features.best_combo, card_ids=features.combo_cards)]
    return BalanceReport(
        combos=combos,
        power_assessment=PowerAssessment(summary=f"{features.power_vs_cost}; real drawback: {features.real_drawback}",
                                         comparable_ids=features.comparable_cards),
        risk_rationale=f"Rule: {why}. {features.rationale} Features: {features.model_dump_json()}",
        risk=risk,
        # The rule gives no probability; the confidence is not meaningful for this judge.
        confidence=0.5,
    )


def free_judge(llm: BaseChatModel, card: CardInput, fmt: str = "modern") -> tuple[BalanceReport, int]:
    """Ask the LLM for a BalanceReport from the card alone, with no citation rules. Nothing is discarded."""
    result = llm.with_structured_output(BalanceReport).invoke([
        SystemMessage(content=_FREE_SYSTEM.format(format=fmt.title())),
        HumanMessage(content=f"Card:\n{describe(card)}"),
    ])
    report = result if isinstance(result, BalanceReport) else BalanceReport.model_validate(result)
    return report, 0


_STRICT_CITES = """Cite the bracket labels exactly (C1, R2). Every claim must cite the hits it relies on.
Claims with no valid citation are discarded. Do not invent rules or cards that are not in the hits."""

# The hybrid judge sees the hits but may also rely on what the model knows, checked against the catalog.
_HYBRID_CITES = """Cite the bracket labels exactly (C1, R2) for the hits you rely on. You may also rely on your
own knowledge of existing Magic cards and of the Comprehensive Rules: name such cards by their exact
English name and such rules by their number (702.19a). Cards you name must exist and be legal in
{format}; any other name is discarded, and a claim left with no valid citation is discarded too."""

assert _STRICT_CITES in _SYSTEM
_HYBRID_SYSTEM = _SYSTEM.replace(_STRICT_CITES, _HYBRID_CITES)

_RULE_NUMBER = re.compile(r"^\d{3}(\.\d+[a-z]?)?$")


def validate_hybrid(report: BalanceReport, labels: dict[str, str], legal: set[str],
                    own: frozenset[str] = frozenset()) -> tuple[BalanceReport, int]:
    """Like validate_citations, but a citation may also be a legal card's name or a rule number.

    Hit labels become hit ids; names and rule numbers are kept as written. The card naming itself
    (`own`) counts as no citation. Returns the cleaned report and how many claims were discarded.
    """
    def cards(citations: list[str]) -> list[str]:
        kept = _valid(citations, labels, "C")
        for citation in citations:
            name = citation.strip()
            if name.casefold() in legal and name.casefold() not in own and name not in kept:
                kept.append(name)
        return kept

    def rules(citations: list[str]) -> list[str]:
        kept = _valid(citations, labels, "R")
        kept += [c.strip() for c in citations if _RULE_NUMBER.match(c.strip()) and c.strip() not in kept]
        return kept

    issues = [issue.model_copy(update={"rule_ids": rules(issue.rule_ids)})
              for issue in report.wording_issues if rules(issue.rule_ids)]
    combos = [combo.model_copy(update={"card_ids": cards(combo.card_ids)})
              for combo in report.combos if cards(combo.card_ids)]
    power = report.power_assessment.model_copy(
        update={"comparable_ids": cards(report.power_assessment.comparable_ids)})
    dropped = len(report.wording_issues) - len(issues) + len(report.combos) - len(combos)
    return report.model_copy(update={"wording_issues": issues, "combos": combos, "power_assessment": power}), dropped


def judge(llm: BaseChatModel, card: CardInput, hits: list[Hit], fmt: str = "modern",
          legal: set[str] | None = None) -> tuple[BalanceReport, int]:
    """Ask the LLM for a BalanceReport, then keep only the claims that cite real hits.

    With `legal` (the hybrid judge) the model may also cite cards and rules it knows; named cards
    must be in `legal`.
    """
    labels, numbered = number_hits(hits)
    prompt = f"Card:\n{describe(card)}\n\nHits:\n{numbered}"
    system = (_SYSTEM if legal is None else _HYBRID_SYSTEM).format(format=fmt.title())
    if any(hit.purpose == "combo" for hit in hits):
        system += _SPLIT_NOTE
    result = llm.with_structured_output(BalanceReport).invoke([
        SystemMessage(content=system),
        HumanMessage(content=prompt),
    ])
    report = result if isinstance(result, BalanceReport) else BalanceReport.model_validate(result)
    if legal is None:
        return validate_citations(report, labels)
    return validate_hybrid(report, labels, legal, frozenset(card.names()))


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
    # cite: true (claims must cite hits), false (no citation rules), catalog (named cards must be legal
    # ones) or hybrid (with retrieval: cite hits, or name legal cards and rule numbers from memory).
    cite = cfg.get("cite", True)
    if cite not in (True, False, "catalog", "hybrid"):
        raise ValueError(f"cite must be true, false, catalog or hybrid, got {cite!r}")
    decide = str(cfg.get("decide", "llm"))
    if decide not in ("llm", "rule"):
        raise ValueError(f"decide must be llm or rule, got {decide!r}")
    retrieves = cfg.get("retrieve", True)
    if (cite in (False, "catalog") or decide == "rule") and retrieves:
        raise ValueError("cite: false|catalog and decide: rule are only for the judge without retrieval (retrieve: false)")
    if cite == "hybrid" and not retrieves:
        raise ValueError("cite: hybrid needs retrieval; without it use cite: catalog")
    legal_by_format: dict[str, set[str]] = {}

    def legal(fmt: str) -> set[str]:
        if fmt not in legal_by_format:
            legal_by_format[fmt] = legal_names(_processed_dir(cfg) / "cards.jsonl", fmt)
        return legal_by_format[fmt]

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
        fmt = state.get("format") or "modern"
        if decide == "rule" or cite in (False, "catalog"):
            if decide == "rule":
                report, dropped = feature_judge(llm, state["card"], fmt), 0
            else:
                report, dropped = free_judge(llm, state["card"], fmt)
            if cite == "catalog":
                own = frozenset(name.casefold() for name in state["card"].names())
                report, dropped = validate_names(report, legal(fmt), own)
            return {"report": report, "dropped_claims": dropped}
        hits = (state.get("oracle_hits") or []) + judge_context(state.get("hits") or [], context_cards, context_rules)
        report, dropped = judge(llm, state["card"], hits, fmt, legal(fmt) if cite == "hybrid" else None)
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
