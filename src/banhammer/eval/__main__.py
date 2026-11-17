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
from banhammer.eval.anonymize import anonymize
from banhammer.eval.metrics import (balanced_accuracy, ban_detected, card_names, false_alarm, false_ban, mean,
                                     ordinal_error, pieces_named, recall_at_k, risk_matches, wilson_interval)
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


def group_of(row: dict) -> str:
    """The gold group of a card: format and expected risk, plus why it was banned when the row says."""
    group = f"{row.get('format', 'modern')} {row['expected_risk']}"
    return f"{group} ({row['ban_reason']})" if row.get("ban_reason") else group


VERDICT_KEYS = ("risk", "risk_match", "false_alarm", "ban_detected", "false_ban", "ordinal_error", "pieces_named")


def verdict_scores(row: dict, report: BalanceReport | None) -> dict:
    """Scores of one report against its gold row. All None when the pipeline gave no report."""
    if report is None:
        return dict.fromkeys(VERDICT_KEYS)
    expected = row["expected_risk"]
    pieces = row.get("combo_pieces") or []
    return {
        "risk": report.risk,
        "risk_match": risk_matches(report, expected),
        "false_alarm": false_alarm(report, expected),
        "ban_detected": ban_detected(report, expected),
        "false_ban": false_ban(report, expected),
        "ordinal_error": ordinal_error(report, expected),
        "pieces_named": pieces_named(report, pieces) if pieces else None,
    }


def score_card(graph, row: dict, embedding_model: str, anonymized: bool = True) -> dict:
    """Run one gold card. `anonymized` sends it as \"Test Card\" so the judge cannot recall it by name."""
    original = CardInput.model_validate(row["card"])
    card = anonymize(original) if anonymized else original
    fmt = row.get("format", "modern")
    with get_usage_metadata_callback() as usage:
        result = graph.invoke({"card": card, "format": fmt, "queries": [], "hits": []})
    cost = run_cost(usage.usage_metadata, result.get("queries") or [], embedding_model)
    # Final card order, after fusion and reranking; recall@25 covers the reranker's candidates.
    hits = [hit if isinstance(hit, Hit) else Hit.model_validate(hit) for hit in result.get("hits") or []]
    report = result.get("report")
    if report is not None and not isinstance(report, BalanceReport):
        report = BalanceReport.model_validate(report)
    pieces = row.get("combo_pieces") or []
    return {
        "card": original.name,
        "format": fmt,
        "group": group_of(row),
        "expected_risk": row["expected_risk"],
        "ban_reason": row.get("ban_reason"),
        **verdict_scores(row, report),
        "recall@5": recall_at_k(hits, pieces, 5) if pieces else None,
        "recall@10": recall_at_k(hits, pieces, 10) if pieces else None,
        "recall@25": recall_at_k(hits, pieces, 25) if pieces else None,
        "dropped_claims": result.get("dropped_claims", 0),
        **cost,
        "retrieved_cards": card_names(hits)[:10],
        "report": None if report is None else report.model_dump(),
    }


def summarize_groups(cards: list[dict]) -> dict[str, dict]:
    """Per gold group: how many cards, the verdicts given, and mean recall@10."""
    groups: dict[str, dict] = {}
    for card in cards:
        group = groups.setdefault(card["group"], {"n": 0, "ok": 0, "watch": 0, "likely_ban": 0, "recalls": []})
        group["n"] += 1
        if card["risk"]:
            group[card["risk"]] += 1
        group["recalls"].append(card["recall@10"])
    for group in groups.values():
        group["recall@10"] = mean(group.pop("recalls"))
    return groups


def _caught(cards: list[dict], reason: str) -> list[bool]:
    return [c["ban_detected"] for c in cards if c["ban_reason"] == reason and c["ban_detected"] is not None]


def summarize(config: str, anonymized: bool, cards: list[dict]) -> dict:
    """Aggregate scores of one config.

    The headline is binary, banned or not, on the bans a card's text can explain (`ban_reason: text`).
    Bans for metagame reasons are reported apart: they depend on the rest of the format, not on the text.
    """
    costs = [c["cost_usd"] for c in cards]
    judged = [c["risk_match"] for c in cards if c["risk_match"] is not None]
    text_bans, meta_bans = _caught(cards, "text"), _caught(cards, "meta")
    false_bans = [c["false_ban"] for c in cards if c["false_ban"] is not None]
    return {
        "config": config,
        "anonymized": anonymized,
        "balanced_accuracy": balanced_accuracy(mean(text_bans), mean(false_bans)),
        "text_bans_caught": [sum(text_bans), len(text_bans)],
        "text_bans_caught_ci95": wilson_interval(sum(text_bans), len(text_bans)),
        "meta_bans_caught": [sum(meta_bans), len(meta_bans)],
        "false_bans": [sum(false_bans), len(false_bans)],
        "false_bans_ci95": wilson_interval(sum(false_bans), len(false_bans)),
        "ordinal_error": mean([c["ordinal_error"] for c in cards]),
        "pieces_named": mean([c["pieces_named"] for c in cards]),
        "recall@5": mean([c["recall@5"] for c in cards]),
        "recall@10": mean([c["recall@10"] for c in cards]),
        "recall@25": mean([c["recall@25"] for c in cards]),
        "risk_accuracy": mean([c["risk_match"] for c in cards]),
        "risk_accuracy_ci95": wilson_interval(sum(judged), len(judged)),
        "false_alarm_rate": mean([c["false_alarm"] for c in cards]),
        "ban_detection_rate": mean([c["ban_detected"] for c in cards]),
        "dropped_per_card": mean([c["dropped_claims"] for c in cards]),
        "input_tokens_per_card": mean([c["input_tokens"] for c in cards]),
        "output_tokens_per_card": mean([c["output_tokens"] for c in cards]),
        "cost_per_card_usd": None if None in costs else mean(costs),
        "cost_total_usd": None if None in costs else sum(costs),
        "cards": len(cards),
        "groups": summarize_groups(cards),
    }


def rescore(run_path: Path, gold_path: Path, out_dir: Path) -> dict:
    """Score a saved run again from its reports, without calling any API.

    Recall and cost come from the run as saved; every verdict score is recomputed against the gold set.
    """
    run = json.loads(run_path.read_text(encoding="utf-8"))
    gold = {row["card"]["name"]: row for row in load_gold(gold_path)}
    cards = []
    for card in run["cards"]:
        row = gold[card["card"]]
        report = None if card["report"] is None else BalanceReport.model_validate(card["report"])
        cards.append({**card, "group": group_of(row), "expected_risk": row["expected_risk"],
                      "ban_reason": row.get("ban_reason"), **verdict_scores(row, report)})
    summary = summarize(run_path.stem, run["summary"].get("anonymized", False), cards)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / run_path.name
    out_path.write_text(json.dumps({"summary": summary, "cards": cards}, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"Wrote {out_path}")
    return summary


def score_config(config_path: Path, gold_path: Path, out_dir: Path, retrieval_only: bool = False,
                 anonymized: bool = True) -> dict:
    """Score every gold card. `retrieval_only` skips the judge and the cards without combo pieces."""
    graph = build_graph(config_path, judge_report=not retrieval_only)
    embedding_model = load_yaml(config_path)["embeddings"].get("model") or "text-embedding-3-small"
    rows = load_gold(gold_path)
    if retrieval_only:
        rows = [row for row in rows if row.get("combo_pieces")]
    cards = []
    for row in rows:
        cards.append(score_card(graph, row, embedding_model, anonymized))
        outcome = cards[-1]['risk'] or f"recall@25 {cards[-1]['recall@25']:.2f}"
        print(f"{config_path.stem}: {cards[-1]['card']} -> {outcome} (${_usd(cards[-1]['cost_usd'])})")
    summary = summarize(config_path.stem, anonymized, cards)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "-retrieval" if retrieval_only else ""
    out_path = out_dir / f"{config_path.stem}{suffix}.json"
    out_path.write_text(json.dumps({"summary": summary, "cards": cards}, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"Wrote {out_path}")
    return summary


COLUMNS = ("balanced_accuracy", "text_bans_caught", "false_bans", "meta_bans_caught", "ordinal_error",
           "pieces_named", "recall@10", "recall@25", "dropped_per_card")
HEADERS = ("ban/legal bal. acc", "text bans caught (95% CI)", "legal→ban (95% CI)", "meta bans caught",
           "ordinal err", "pieces named", "recall@10", "recall@25", "dropped")
COST_COLUMNS = ("input_tokens_per_card", "output_tokens_per_card", "cost_per_card_usd", "cost_total_usd")
COST_HEADERS = ("in tok/card", "out tok/card", "usd/card", "usd total")


def _usd(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _cell(row: dict, column: str) -> str:
    """A metric as text: counts as "2/5", with their 95% interval when the summary has one."""
    value = row.get(column)
    if value is None:
        return "n/a"
    if not isinstance(value, list):
        return f"{value:.3f}"
    text = f"{value[0]}/{value[1]}"
    interval = row.get(f"{column}_ci95")
    return f"{text} ({interval[0]:.2f}-{interval[1]:.2f})" if interval else text


def render_table(rows: list[dict]) -> None:
    table = Table(title="Banhammer eval (a constant answer scores 0.5 balanced accuracy)")
    table.add_column("config")
    for header in HEADERS:
        table.add_column(header, justify="right")
    table.add_column("cards", justify="right")
    for row in rows:
        cells = [_cell(row, column) for column in COLUMNS]
        table.add_row(row["config"], *cells, str(row["cards"]))
    console = Console()
    console.print(table)

    groups = Table(title="Verdicts per gold group (ok / watch / likely_ban) and recall@10")
    groups.add_column("config")
    groups.add_column("group")
    groups.add_column("n", justify="right")
    groups.add_column("ok / watch / likely_ban", justify="right")
    groups.add_column("recall@10", justify="right")
    for row in rows:
        for name, group in sorted(row["groups"].items()):
            recall = "n/a" if group["recall@10"] is None else f"{group['recall@10']:.2f}"
            groups.add_row(row["config"], name, str(group["n"]),
                           f"{group['ok']} / {group['watch']} / {group['likely_ban']}", recall)
    console.print(groups)

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
    runs = parser.add_mutually_exclusive_group(required=True)
    runs.add_argument("--configs", nargs="+", type=Path)
    runs.add_argument("--rescore", nargs="+", type=Path, metavar="RUN",
                      help="score saved run files again against the gold set, without calling any API")
    parser.add_argument("--gold", type=Path, default=ROOT / "data" / "eval" / "gold.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "eval")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="skip the judge and the cards without combo pieces; only recall is scored")
    parser.add_argument("--keep-names", action="store_true",
                        help="send gold cards with their real names (by default they are sent as \"Test Card\")")
    args = parser.parse_args()
    if args.rescore:
        render_table([rescore(path, args.gold, args.out) for path in args.rescore])
        return
    load_env()
    render_table([score_config(path, args.gold, args.out, args.retrieval_only, not args.keep_names)
                  for path in args.configs])


if __name__ == "__main__":
    main()
