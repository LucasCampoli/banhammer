"""Run the gold set through one or more pipeline configs, print a comparison and save the runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from banhammer.shared.config import ROOT, load_env
from banhammer.eval.metrics import card_names, false_alarm, mean, recall_at_k, risk_matches
from banhammer.models import BalanceReport, CardInput, Hit
from banhammer.pipeline import build_graph


def load_gold(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score_card(graph, row: dict) -> dict:
    card = CardInput.model_validate(row["card"])
    fmt = row.get("format", "modern")
    result = graph.invoke({"card": card, "format": fmt, "queries": [], "hits": []})
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
        "retrieved_cards": card_names(hits)[:10],
        "report": report.model_dump(),
    }


def score_config(config_path: Path, gold_path: Path, out_dir: Path) -> dict:
    graph = build_graph(config_path)
    cards = []
    for row in load_gold(gold_path):
        cards.append(score_card(graph, row))
        print(f"{config_path.stem}: {cards[-1]['card']} -> {cards[-1]['risk']}")
    summary = {
        "config": config_path.stem,
        "recall@5": mean([c["recall@5"] for c in cards]),
        "recall@10": mean([c["recall@10"] for c in cards]),
        "recall@25": mean([c["recall@25"] for c in cards]),
        "risk_accuracy": mean([c["risk_match"] for c in cards]),
        "false_alarm_rate": mean([c["false_alarm"] for c in cards]),
        "dropped_per_card": mean([c["dropped_claims"] for c in cards]),
        "cards": len(cards),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{config_path.stem}.json"
    out_path.write_text(json.dumps({"summary": summary, "cards": cards}, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"Wrote {out_path}")
    return summary


COLUMNS = ("recall@5", "recall@10", "recall@25", "risk_accuracy", "false_alarm_rate", "dropped_per_card")


def render_table(rows: list[dict]) -> None:
    table = Table(title="Banhammer eval")
    table.add_column("config")
    for column in COLUMNS:
        table.add_column(column, justify="right")
    table.add_column("cards", justify="right")
    for row in rows:
        cells = ["n/a" if row[c] is None else f"{row[c]:.3f}" for c in COLUMNS]
        table.add_row(row["config"], *cells, str(row["cards"]))
    Console().print(table)


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
