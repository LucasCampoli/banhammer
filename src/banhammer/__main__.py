"""CLI: python -m banhammer evaluate <card.json>"""
from __future__ import annotations

import argparse
from pathlib import Path

from banhammer.shared.config import ROOT, load_env
from banhammer.models import BalanceReport, CardInput
from banhammer.pipeline import build_graph
from banhammer.shared.report import render_markdown, report_stem


def evaluate(card_path: Path, config_path: Path, out_dir: Path) -> Path:
    load_env()
    card = CardInput.model_validate_json(card_path.read_text(encoding="utf-8"))
    result = build_graph(config_path).invoke({"card": card, "queries": [], "hits": []})
    report = result["report"]
    if not isinstance(report, BalanceReport):
        report = BalanceReport.model_validate(report)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = report_stem(card.name)
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(card, report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m banhammer",
        description="Evaluate a card. Ingest, index and eval are separate modules.",
        epilog="Also: python -m banhammer.ingest, python -m banhammer.index, python -m banhammer.eval",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = sub.add_parser("evaluate", help="run the pipeline on a card JSON file")
    evaluate_parser.add_argument("card", type=Path)
    evaluate_parser.add_argument("--config", type=Path, default=ROOT / "configs" / "pipeline.yaml")
    evaluate_parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "reports")
    args = parser.parse_args()
    if args.command == "evaluate":
        evaluate(args.card, args.config, args.out)


if __name__ == "__main__":
    main()
