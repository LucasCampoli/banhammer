"""Sample the OCR benchmark from the Scryfall bulk: a uniform sample plus a few cards per stratum.

Run it once and commit the resulting id list; rerunning with another bulk file
or seed changes the benchmark and makes old results incomparable.
"""
from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from banhammer.ocr.benchmark import (
    DEFAULT_BENCHMARK, DEFAULT_BULK, STRATA, UNIFORM, load_population, write_benchmark,
)


def sample(population: list[dict], uniform: int, per_stratum: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    picked: dict[str, str] = {card["id"]: UNIFORM for card in rng.sample(population, uniform)}
    for stratum, matches in STRATA.items():
        candidates = [card for card in population if matches(card) and card["id"] not in picked]
        for card in rng.sample(candidates, min(per_stratum, len(candidates))):
            picked[card["id"]] = stratum
    oracle_ids = {card["id"]: card["oracle_id"] for card in population}
    return [{"id": cid, "oracle_id": oracle_ids[cid], "stratum": stratum} for cid, stratum in picked.items()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bulk", type=Path, default=DEFAULT_BULK, help="Scryfall oracle_cards bulk (.jsonl or .jsonl.gz)")
    ap.add_argument("--out", type=Path, default=DEFAULT_BENCHMARK)
    ap.add_argument("--uniform", type=int, default=150, help="cards drawn uniformly from the OCR population")
    ap.add_argument("--per-stratum", type=int, default=6, help="extra cards drawn from each stratum")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    population = load_population(args.bulk)
    entries = sample(population, args.uniform, args.per_stratum, args.seed)
    write_benchmark(args.out, entries)
    print(f"OCR population: {len(population)} cards")
    print(f"Wrote {len(entries)} cards to {args.out}: {dict(Counter(e['stratum'] for e in entries))}")


if __name__ == "__main__":
    main()
