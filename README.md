# Banhammer

Balance auditor and rules arbiter for Magic: The Gathering cards. The idea: take an image of a new card, read its text with OCR, then use RAG over the Comprehensive Rules and the Scryfall catalog to find wording conflicts and broken combos. Target format is Modern; I prototype with Standard cards.

## Status

Only the OCR stage works for now. `parsing/`, `retrieval/` and `verdict/` are empty.

Mana costs are read by template matching, including hybrid and Phyrexian symbols. Symbols inside the rules text (`{T}`, `{2}{R}`) are not handled yet, and the OCR also picks up flavor text, which hurts the `rules_text` score.

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

## Usage

```bash
python scripts/download_samples.py          # ~30 cards into images/samples/
python scripts/download_mana_symbols.py     # symbol templates into data/mana_templates/
python -m banhammer.ocr images/samples/<card>.jpg --debug
python scripts/evaluate_ocr.py              # table + outputs/ocr_eval.csv
pytest
```

`--debug` saves the zone overlay and the preprocessed crops to `outputs/debug/<card>/`. Use `--photo` for real photos (perspective correction).

Zone coordinates are in `configs/zones.yaml` (fractions of the 672x936 card) and the OCR settings in `configs/ocr.yaml`.

## Notes

Card images and text belong to Wizards of the Coast, so `data/`, `images/` and `outputs/` are gitignored. Parts of the code were written with Claude Code.
