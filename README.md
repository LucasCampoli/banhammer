# Banhammer

Balance auditor and rules arbiter for Magic: The Gathering cards. The idea: take an image of a new card, read its text with OCR, then use RAG over the Comprehensive Rules and the Scryfall catalog to find wording conflicts and broken combos. Target format is Modern; I prototype with Standard cards.

## Status

Phase 1 is a naive RAG pipeline: the whole card text is embedded, similar Modern cards and Comprehensive Rules are retrieved from Pinecone, the card itself is dropped from the hits, and an LLM writes a balance report. OCR still works as a separate stage. Symbols inside the rules text (`{T}`, `{2}{R}`) are not handled yet, and the OCR also picks up flavor text, which hurts the `rules_text` score.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the pipeline and the later phases.

## Setup

Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Tesseract is a separate program, pip doesn't install it:

- Linux: `sudo apt install tesseract-ocr`
- macOS: `brew install tesseract`
- Windows: installer from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki), then add it to `PATH` or set `TESSERACT_CMD` to the `.exe`.

Copy [.env.example](.env.example) to `.env` and set `BANHAMMER_CONTACT`, `OPENAI_API_KEY` and `PINECONE_API_KEY`.

## Usage

```bash
python -m banhammer.ingest                  # Scryfall bulk + Comprehensive Rules -> data/processed/
python -m banhammer.index                   # embed data/processed into Pinecone (cards, rules)
python -m banhammer evaluate path/to/card.json
python -m banhammer.eval --configs configs/pipeline.yaml

python scripts/download_samples.py          # ~30 cards into images/samples/
python scripts/download_mana_symbols.py     # symbol templates into data/mana_templates/
python -m banhammer.ocr images/samples/<card>.jpg --debug
python scripts/evaluate_ocr.py              # table + outputs/ocr_eval.csv
pytest
```

`evaluate` reads a Scryfall-shaped card JSON and writes `outputs/reports/<card name>.json` and `.md`. `ingest` skips a source whose raw or processed file is already present; pass `--force` to refresh it. The embedding model in `configs/ingest.yaml` and `configs/pipeline.yaml` has to be the same one.

`--debug` saves the zone overlay and the preprocessed crops to `outputs/debug/<card>/`. Use `--photo` for real photos (perspective correction).

Zone coordinates are in `configs/zones.yaml` (fractions of the 672x936 card) and the OCR settings in `configs/ocr.yaml`.

## Notes

Card images and downloaded card data belong to Wizards of the Coast, so `data/raw`, `data/processed`, `images/` and `outputs/` are gitignored. `data/eval` is the hand-written gold set and is versioned. Parts of the code were written with Claude Code.
