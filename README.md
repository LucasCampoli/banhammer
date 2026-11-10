# Banhammer

Balance auditor and rules arbiter for Magic: The Gathering cards. The idea: take a new card, then use RAG over the Comprehensive Rules and the Scryfall catalog to find wording conflicts and broken combos. A card is audited against Modern or Standard. An OCR stage can read the card from an image.

## Status

Version 2 (stage 2) adds advanced RAG behind config flags, measured one technique at a time: decomposition of the card into primitives and query expansion toward combo counterparts, hybrid dense + BM25 search with reciprocal rank fusion, type and mana value filters, and reranking with an LLM or a cross-encoder. With every flag off it is version 1.

Version 1 (stage 1): the pipeline embeds the whole card text, retrieves similar cards from the audited format and Comprehensive Rules from Pinecone, drops the card itself from the hits, and asks an LLM for a balance report. Claims in the report that do not cite a retrieved rule or card are discarded. OCR is a separate stage. Symbols inside the rules text (`{T}`, `{2}{R}`) are not handled yet, and the OCR also picks up flavor text, which hurts the `rules_text` score.

See [ARCHITECTURE.md](ARCHITECTURE.md) for how the pieces fit together.

## Setup

Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"         # add ,rerank for the cross-encoder reranker (pulls in torch)
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
python -m banhammer evaluate path/to/card.json --format modern   # or standard
python -m banhammer.eval --configs configs/naive.yaml configs/pipeline.yaml
python -m banhammer.eval --configs configs/naive.yaml configs/pipeline.yaml configs/ablation/*.yaml   # full ablation
python -m banhammer evaluate path/to/card.json --format modern --config configs/ablation/4_rerank_llm.yaml

python scripts/download_samples.py          # ~30 cards into images/samples/
python scripts/download_mana_symbols.py     # symbol templates into data/mana_templates/
python -m banhammer.ocr images/samples/<card>.jpg --debug
python scripts/evaluate_ocr.py              # table + outputs/ocr_eval.csv
pytest
```

`evaluate` reads a Scryfall-shaped card JSON and writes `outputs/reports/<card name>.json` and `.md`. `ingest` skips a source whose raw or processed file is already present; pass `--force` to refresh it. Data processed before stage 1 lacks the `formats` metadata, so run `python -m banhammer.ingest --force` and `python -m banhammer.index` again.

`banhammer.eval` prints one row per config and writes every card's result to `outputs/eval/<config>.json`. `configs/naive.yaml` is the baseline without the format filter; `configs/pipeline.yaml` is version 1; `configs/ablation/` adds one stage 2 technique per file (see [ARCHITECTURE.md](ARCHITECTURE.md#ablation)). The configs with `hybrid: true` build BM25 from `data/processed/cards.jsonl`, so `python -m banhammer.ingest` has to run first on a new machine (`data/` is not in git). `python -m banhammer.index` is only needed if your Pinecone project does not have the index yet. The embedding model in `configs/ingest.yaml` and `configs/pipeline.yaml` has to be the same one.

`--debug` saves the zone overlay and the preprocessed crops to `outputs/debug/<card>/`. Use `--photo` for real photos (perspective correction).

Zone coordinates are in `configs/zones.yaml` (fractions of the 672x936 card) and the OCR settings in `configs/ocr.yaml`.

## Results

Stage 2 ablation on the 37-card gold set (`data/eval/gold.jsonl`). Not run yet: the course LiteLLM proxy was returning `429 no credits remaining`. Fill it in from the table `banhammer.eval` prints.

| # | Config | recall@5 | recall@10 | recall@25 | risk_accuracy | false_alarm_rate | dropped_per_card |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | `naive` | | | | | | |
| 1 | `pipeline` | | | | | | |
| 2 | `ablation/1_decompose` | | | | | | |
| 3 | `ablation/2_hybrid` | | | | | | |
| 4 | `ablation/3_type_mana_filter` | | | | | | |
| 5 | `ablation/4_rerank_llm` | | | | | | |
| 6 | `ablation/5_rerank_cross_encoder` | | | | | | |

## Notes

Card images and downloaded card data belong to Wizards of the Coast, so `data/raw`, `data/processed`, `images/` and `outputs/` are gitignored. `data/eval` is the hand-written gold set and is versioned. Parts of the code were written with Claude Code.
