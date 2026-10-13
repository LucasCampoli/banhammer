"""Run the OCR on images/samples and compare with ground_truth.json (CER + exact match per field)."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from rich.console import Console
from rich.table import Table

from banhammer.ocr.pipeline import OCRPipeline

FIELDS = ["name", "mana_cost", "type_line", "rules_text", "power", "toughness"]
GT_KEY = {"rules_text": "oracle_text"}


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def cer(pred: str, ref: str) -> float:
    return levenshtein(pred, ref) / max(len(ref), 1) if ref else float(len(pred) > 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", type=Path, default=Path("images/samples"))
    ap.add_argument("--csv", type=Path, default=Path("outputs/ocr_eval.csv"))
    args = ap.parse_args()
    truth = json.loads((args.samples / "ground_truth.json").read_text())
    pipe = OCRPipeline()
    rows, stats = [], {f: {"cer": [], "exact": []} for f in FIELDS}
    for t in truth:
        card = pipe.run(args.samples / f"{t['id']}.jpg")
        row = {"id": t["id"], "name_gt": t["name"]}
        for f in FIELDS:
            ref = norm(t.get(GT_KEY.get(f, f)))
            pred = norm(getattr(card, f))
            if f == "rules_text":  # compare ignoring paragraph breaks
                ref, pred = norm(ref.replace("\n", " ")), norm(pred.replace("\n", " "))
            c, ex = cer(pred, ref), pred == ref
            stats[f]["cer"].append(c)
            stats[f]["exact"].append(ex)
            row.update({f"{f}_pred": pred, f"{f}_ref": ref, f"{f}_cer": round(c, 4)})
        rows.append(row)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    table = Table(title=f"OCR evaluation ({len(truth)} cards)")
    for col in ("field", "mean CER", "exact match %"):
        table.add_column(col)
    for f in FIELDS:
        n = len(stats[f]["cer"])
        table.add_row(f, f"{sum(stats[f]['cer']) / n:.3f}", f"{100 * sum(stats[f]['exact']) / n:.1f}")
    Console().print(table)
    print(f"Per-card CSV: {args.csv}")


if __name__ == "__main__":
    main()
