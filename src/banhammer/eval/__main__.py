"""Run the gold set through one or more pipeline configs and print a comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from banhammer.shared.config import ROOT, load_env
from banhammer.eval.metrics import recall_at_k, risk_matches
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


def score_config(config_path: Path, gold_path: Path, k: int) -> dict:
    graph = build_graph(config_path)
    recalls: list[float] = []
    risks: list[bool] = []
    for row in load_gold(gold_path):
        card = CardInput.model_validate(row["card"])
        result = graph.invoke({"card": card, "queries": [], "hits": []})
        hits = [_as_hit(hit) for hit in result.get("hits") or []]
        report = result["report"]
        if not isinstance(report, BalanceReport):
            report = BalanceReport.model_validate(report)
        pieces = row.get("combo_pieces") or []
        if pieces:
            recalls.append(recall_at_k(hits, pieces, k))
        risks.append(risk_matches(report, row["expected_risk"]))
    return {
        "config": str(config_path),
        "recall": sum(recalls) / len(recalls) if recalls else None,
        "risk_accuracy": sum(risks) / len(risks) if risks else None,
        "cards": len(risks),
    }


def _as_hit(hit: Hit | dict) -> Hit:
    return hit if isinstance(hit, Hit) else Hit.model_validate(hit)


def render_table(rows: list[dict]) -> None:
    table = Table(title="Banhammer eval")
    table.add_column("config")
    table.add_column("recall@10", justify="right")
    table.add_column("risk accuracy", justify="right")
    table.add_column("cards", justify="right")
    for row in rows:
        recall = "n/a" if row["recall"] is None else f"{row['recall']:.3f}"
        accuracy = "n/a" if row["risk_accuracy"] is None else f"{row['risk_accuracy']:.3f}"
        table.add_row(row["config"], recall, accuracy, str(row["cards"]))
    Console().print(table)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m banhammer.eval", description=__doc__)
    parser.add_argument("--configs", nargs="+", type=Path, required=True)
    parser.add_argument("--gold", type=Path, default=ROOT / "data" / "eval" / "gold.jsonl")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    load_env()
    render_table([score_config(path, args.gold, args.k) for path in args.configs])


if __name__ == "__main__":
    main()
