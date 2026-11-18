"""How many gold cards the judge model recognizes from their anonymized text.

Each card goes to the model exactly as the judge sees it ("Test Card"), with one question: which
existing card is this? A right guess means the judge can answer from memory instead of reasoning,
so its verdict on that card says little about how it would do with a new design.

    python -m banhammer.eval.memorization --config configs/no_rag.yaml [--gold data/eval/gold.jsonl]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from banhammer.eval.__main__ import group_of, load_gold
from banhammer.eval.anonymize import anonymize
from banhammer.models import CardInput
from banhammer.pipeline.graph import describe
from banhammer.providers import get_llm
from banhammer.shared.config import ROOT, load_env, load_yaml

_SYSTEM = """You are an expert on Magic: The Gathering cards.
You are given the rules text of a card with its name hidden as "Test Card".
If it is an existing card you recognize, give its exact English name. If you do not recognize it,
answer "unknown". Do not guess a card unless you are fairly sure."""


class Guess(BaseModel):
    name: str = Field(description='The exact English name of the card, or "unknown".')
    confidence: float = Field(ge=0, le=1)


def recognized(guess: str, real: str) -> bool:
    """A guess counts when it names the card: the full name, or the part before the comma."""
    guess, real = guess.strip().casefold(), real.casefold()
    return bool(guess) and (guess == real or guess == real.split(",")[0])


def probe(llm, row: dict) -> dict:
    card = CardInput.model_validate(row["card"])
    answer = llm.with_structured_output(Guess).invoke([
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=f"Card:\n{describe(anonymize(card))}"),
    ])
    guess = answer if isinstance(answer, Guess) else Guess.model_validate(answer)
    return {"card": card.name, "group": group_of(row), "guess": guess.name, "confidence": guess.confidence,
            "recognized": recognized(guess.name, card.name)}


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m banhammer.eval.memorization", description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "no_rag.yaml")
    parser.add_argument("--gold", type=Path, default=ROOT / "data" / "eval" / "gold.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "eval" / "memorization.json")
    args = parser.parse_args()
    load_env()
    llm = get_llm(load_yaml(args.config)["llm"])
    rows = [probe(llm, row) for row in load_gold(args.gold)]
    groups: dict[str, list[bool]] = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row["recognized"])
        mark = "RECOGNIZED" if row["recognized"] else ""
        print(f"{row['card']:<35} -> {row['guess']:<35} {row['confidence']:.2f} {mark}")
    summary = {"recognized": sum(r["recognized"] for r in rows), "cards": len(rows),
               "groups": {name: [sum(hits), len(hits)] for name, hits in sorted(groups.items())}}
    print(json.dumps(summary, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "cards": rows}, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
