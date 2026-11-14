"""Run the OCR on the benchmark cards and compare it with Scryfall Oracle text.

Headline numbers use only the uniform sample, so they estimate accuracy on the
whole OCR population. The other rows group every benchmark card that has a
given feature. P/T is scored only on cards that print one.
"""
from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean

from rich.console import Console
from rich.table import Table

from banhammer.ocr.benchmark import (
    DEFAULT_BENCHMARK, DEFAULT_BULK, DEFAULT_IMAGES, STRATA, UNIFORM, cards_by_id, read_benchmark,
)
from banhammer.ocr.evaluation import cer, flatten, normalize_rules, normalize_type_line
from banhammer.ocr.pipeline import DEFAULT_CONFIG, OCRPipeline, load_config

_pipeline: OCRPipeline | None = None


def _init_worker(config_path: Path) -> None:
    global _pipeline
    _pipeline = OCRPipeline(load_config(config_path))


def _run(image: Path) -> dict:
    return _pipeline.run(image).model_dump()


def score(card: dict, ocr: dict) -> dict:
    name = card["name"]
    rules_ref = normalize_rules(card.get("oracle_text"), name)
    rules_flavor_ref = normalize_rules(f"{card.get('oracle_text') or ''} {card.get('flavor_text') or ''}", name)
    rules_pred = normalize_rules(ocr["rules_text"], name)
    type_ref, type_pred = normalize_type_line(card["type_line"]), normalize_type_line(ocr["type_line"])
    name_pred = flatten(ocr["name"])
    has_pt = card.get("power") is not None
    return {
        "name_pred": name_pred,
        "name_exact": name_pred == name,
        "name_cer": cer(name_pred, name),
        "mana_pred": ocr["mana_cost"],
        "mana_ref": card.get("mana_cost") or "",
        "mana_exact": ocr["mana_cost"] == (card.get("mana_cost") or ""),
        "type_pred": type_pred,
        "type_ref": type_ref,
        "type_exact": type_pred == type_ref,
        "type_cer": cer(type_pred, type_ref),
        "has_pt": has_pt,
        "pt_pred": f"{ocr['power']}/{ocr['toughness']}",
        "pt_ref": f"{card.get('power')}/{card.get('toughness')}" if has_pt else "",
        "pt_exact": (ocr["power"], ocr["toughness"]) == (card.get("power"), card.get("toughness")) if has_pt else None,
        "rules_pred": rules_pred,
        "rules_ref": rules_ref,
        "rules_cer": cer(rules_pred, rules_ref),
        "rules_flavor_cer": cer(rules_pred, rules_flavor_ref),
    }


def percent(values: list[bool]) -> str:
    return f"{100 * mean(values):.1f}" if values else "-"


def summary_row(label: str, rows: list[dict]) -> list[str]:
    pt = [r["pt_exact"] for r in rows if r["has_pt"]]
    return [
        label, str(len(rows)),
        percent([r["name_exact"] for r in rows]),
        percent([r["mana_exact"] for r in rows]),
        percent([r["type_exact"] for r in rows]),
        f"{percent(pt)} ({len(pt)})",
        f"{mean(r['rules_cer'] for r in rows):.3f}",
        f"{mean(r['rules_flavor_cer'] for r in rows):.3f}",
    ]


def groups(rows: list[dict], cards: dict[str, dict]) -> list[tuple[str, list[dict]]]:
    uniform = [r for r in rows if r["stratum"] == UNIFORM]
    out = [
        ("uniform", uniform),
        ("  frame 2015", [r for r in uniform if r["frame"] == "2015"]),
        ("  frame 2003", [r for r in uniform if r["frame"] == "2003"]),
    ]
    for stratum, matches in STRATA.items():
        out.append((stratum, [r for r in rows if matches(cards[r["id"]])]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bulk", type=Path, default=DEFAULT_BULK)
    ap.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    ap.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="OCR config to evaluate")
    ap.add_argument("--csv", type=Path, default=Path("outputs/ocr_eval.csv"))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    entries = read_benchmark(args.benchmark)
    cards = cards_by_id(args.bulk, {e["id"] for e in entries})
    images = [args.images / f"{e['id']}.jpg" for e in entries]
    missing = [p for p in images if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images missing in {args.images}; run scripts/download_benchmark_images.py")

    os.environ["OMP_THREAD_LIMIT"] = "1"  # one Tesseract thread per worker process
    with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(args.config,)) as pool:
        results = list(pool.map(_run, images))

    rows = []
    for entry, ocr in zip(entries, results):
        card = cards[entry["id"]]
        rows.append({"id": entry["id"], "stratum": entry["stratum"], "frame": card["frame"],
                     "name": card["name"], **score(card, ocr)})

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    table = Table(title=f"OCR benchmark ({len(rows)} cards, config {args.config.name})")
    for column in ("group", "n", "name %", "mana %", "type %", "P/T % (n)", "rules CER", "rules+flavor CER"):
        table.add_column(column, justify="left" if column == "group" else "right", no_wrap=True)
    for label, group in groups(rows, cards):
        if group:
            table.add_row(*summary_row(label, group))
    Console().print(table)
    print(f"Per-card CSV: {args.csv}")


if __name__ == "__main__":
    main()
