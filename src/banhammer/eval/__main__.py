"""Run the gold set through one or more pipeline configs, print a comparison and save the runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain_core.callbacks import get_usage_metadata_callback
from rich.console import Console
from rich.table import Table

from banhammer.shared.config import ROOT, load_env, load_yaml
from banhammer.eval.cost import chat_usage, embedding_cost, embedding_tokens
from banhammer.eval.metrics import card_names, false_alarm, mean, recall_at_k, risk_matches
from banhammer.models import BalanceReport, CardInput, Hit, Query
from banhammer.pipeline import build_graph


def load_gold(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_cost(chat: dict, queries: list, embedding_model: str) -> dict:
    """Tokens and dollars of one card: every chat call, plus one embedding per (dense) query."""
    texts = [(query if isinstance(query, Query) else Query.model_validate(query)).text for query in queries]
    emb_tokens = embedding_tokens(texts)
    emb_cost = embedding_cost(emb_tokens, embedding_model)
    usage = chat_usage(chat)
    total = None if usage["chat_cost_usd"] is None or emb_cost is None else usage["chat_cost_usd"] + emb_cost
    return {**usage, "embedding_tokens": emb_tokens, "cost_usd": total}


def score_card(graph, row: dict, embedding_model: str) -> dict:
    card = CardInput.model_validate(row["card"])
    fmt = row.get("format", "modern")
    with get_usage_metadata_callback() as usage:
        result = graph.invoke({"card": card, "format": fmt, "queries": [], "hits": []})
    cost = run_cost(usage.usage_metadata, result.get("queries") or [], embedding_model)
    # Final card order, after fusion and reranking; recall@25 covers the reranker's candidates.
    hits = [hit if isinstance(hit, Hit) else Hit.model_validate(hit) for hit in result.get("hits") or []]
    report = result["report"]
    if not isinstance(report, BalanceReport):
        report = BalanceReport.model_validate(report)
    pieces = row.get("combo_pieces") or []
    expected = row["expected_risk"]
    return {
        "card": card.name,
        "format": fmt,
        "expected_risk": expected,
        "risk": report.risk,
        "risk_match": risk_matches(report, expected),
        "false_alarm": false_alarm(report, expected),
        "recall@5": recall_at_k(hits, pieces, 5) if pieces else None,
        "recall@10": recall_at_k(hits, pieces, 10) if pieces else None,
        "recall@25": recall_at_k(hits, pieces, 25) if pieces else None,
        "dropped_claims": result.get("dropped_claims", 0),
        **cost,
        "retrieved_cards": card_names(hits)[:10],
        "report": report.model_dump(),
    }


def score_config(config_path: Path, gold_path: Path, out_dir: Path) -> dict:
    graph = build_graph(config_path)
    embedding_model = load_yaml(config_path)["embeddings"].get("model") or "text-embedding-3-small"
    cards = []
    for row in load_gold(gold_path):
        cards.append(score_card(graph, row, embedding_model))
        print(f"{config_path.stem}: {cards[-1]['card']} -> {cards[-1]['risk']} (${_usd(cards[-1]['cost_usd'])})")
    costs = [c["cost_usd"] for c in cards]
    summary = {
        "config": config_path.stem,
        "recall@5": mean([c["recall@5"] for c in cards]),
        "recall@10": mean([c["recall@10"] for c in cards]),
        "recall@25": mean([c["recall@25"] for c in cards]),
        "risk_accuracy": mean([c["risk_match"] for c in cards]),
        "false_alarm_rate": mean([c["false_alarm"] for c in cards]),
        "dropped_per_card": mean([c["dropped_claims"] for c in cards]),
        "input_tokens_per_card": mean([c["input_tokens"] for c in cards]),
        "output_tokens_per_card": mean([c["output_tokens"] for c in cards]),
        "cost_per_card_usd": None if None in costs else mean(costs),
        "cost_total_usd": None if None in costs else sum(costs),
        "cards": len(cards),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{config_path.stem}.json"
    out_path.write_text(json.dumps({"summary": summary, "cards": cards}, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"Wrote {out_path}")
    return summary


COLUMNS = ("recall@5", "recall@10", "recall@25", "risk_accuracy", "false_alarm_rate", "dropped_per_card")
COST_COLUMNS = ("input_tokens_per_card", "output_tokens_per_card", "cost_per_card_usd", "cost_total_usd")
COST_HEADERS = ("in tok/card", "out tok/card", "usd/card", "usd total")


def _usd(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def render_table(rows: list[dict]) -> None:
    table = Table(title="Banhammer eval")
    table.add_column("config")
    for column in COLUMNS:
        table.add_column(column, justify="right")
    table.add_column("cards", justify="right")
    for row in rows:
        cells = ["n/a" if row[c] is None else f"{row[c]:.3f}" for c in COLUMNS]
        table.add_row(row["config"], *cells, str(row["cards"]))
    console = Console()
    console.print(table)

    costs = Table(title="Banhammer eval: cost (chat + query embeddings)")
    costs.add_column("config")
    for header in COST_HEADERS:
        costs.add_column(header, justify="right")
    for row in rows:
        tokens = ["n/a" if row[c] is None else f"{row[c]:,.0f}" for c in COST_COLUMNS[:2]]
        costs.add_row(row["config"], *tokens, *(_usd(row[c]) for c in COST_COLUMNS[2:]))
    console.print(costs)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m banhammer.eval", description=__doc__)
    parser.add_argument("--configs", nargs="+", type=Path, required=True)
    parser.add_argument("--gold", type=Path, default=ROOT / "data" / "eval" / "gold.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "eval")
    args = parser.parse_args()
    load_env()
    render_table([score_config(path, args.gold, args.out) for path in args.configs])


if __name__ == "__main__":
    main()
