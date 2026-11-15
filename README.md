# Banhammer

Balance auditor and rules arbiter for Magic: The Gathering cards. The idea: take an image of a new card, read its text with OCR, then use RAG over the Comprehensive Rules and the Scryfall catalog to find wording conflicts and broken combos. Target format is Modern; I prototype with Standard cards.

## Status

The pipeline embeds the whole card text, retrieves similar Modern cards and Comprehensive Rules from Pinecone, drops the card itself from the hits, and asks an LLM for a balance report. OCR is a separate stage. Symbols inside the rules text (`{T}`, `{2}{R}`) are not handled yet, and the OCR also picks up flavor text, which hurts the `rules_text` score.

See [ARCHITECTURE.md](ARCHITECTURE.md) for how the pieces fit together.

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

python scripts/download_tessdata.py         # tessdata_best model into data/tessdata/
python scripts/download_mana_symbols.py     # symbol templates into data/mana_templates/
python scripts/download_type_vocabulary.py  # type line words into data/type_vocabulary.json
python scripts/download_benchmark_images.py # OCR benchmark images into images/benchmark/
python scripts/transcribe_cards.py          # OCR every card in scope into data/processed/ocr_cards.jsonl
python -m banhammer.ocr images/benchmark/<card id>.jpg --debug
python scripts/evaluate_ocr.py              # table + outputs/ocr_eval.csv
pytest
```

`evaluate` reads a Scryfall-shaped card JSON and writes `outputs/reports/<card name>.json` and `.md`. `ingest` skips a source whose raw or processed file is already present; pass `--force` to refresh it. The embedding model in `configs/ingest.yaml` and `configs/pipeline.yaml` has to be the same one.

`--debug` saves the zone overlay and the preprocessed crops to `outputs/debug/<card>/`. Use `--photo` for real photos (perspective correction).

The OCR only transcribes single-faced cards with the 2003 or 2015 frame that are not planeswalkers (`banhammer/ocr/scope.py`, about 94% of the indexed cards); the rest keep their Scryfall text. `transcribe_cards.py` downloads their images into `images/cards/` and takes about two hours with 16 cores; it resumes where it stopped. Ingestion uses a card's transcription when there is one (`source: ocr` in the metadata) and its Scryfall text otherwise, so it also works without running it. `python -m banhammer.ingest --only scryfall_cards --card-text ocr` rebuilds the card records from the transcriptions (and fails if there are none); `--card-text scryfall` rebuilds them with Scryfall text for every card, for example to compare the two indexes. The benchmark in `data/eval/ocr_benchmark.jsonl` lists 150 cards drawn uniformly from that population plus a few per risky feature (2003 frame, legendary, white border, low-resolution scan, ...). `scripts/build_ocr_benchmark.py` made it; do not rebuild it unless you want results that cannot be compared with older runs. The benchmark and evaluation scripts read the Scryfall bulk from `data/raw/oracle-cards.jsonl.gz` (pass `--bulk` for another file). Rules text is compared with Oracle text after removing the most common templating changes, so the error still includes some errata. `data/eval/ocr_gold.jsonl` has 30 of those cards transcribed by hand as printed; the evaluation scores them against both references, and the gap between the two is errata, not reading error.

Zone coordinates are in `configs/zones.yaml` (fractions of the 672x936 card) and the OCR settings in `configs/ocr.yaml`.

## Notes

Card images and downloaded card data belong to Wizards of the Coast, so `data/raw`, `data/processed`, `images/` and `outputs/` are gitignored. `data/eval` is the hand-written gold set and is versioned. Parts of the code were written with Claude Code.
