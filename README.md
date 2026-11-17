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
python -m banhammer.eval --configs configs/ablation/7_decompose_gpt5_mini.yaml --retrieval-only   # recall only, no judge
python -m banhammer.eval --rescore outputs/eval/4_rerank_llm.json   # score a saved run again, no API calls
python -m banhammer evaluate path/to/card.json --format modern --config configs/ablation/4_rerank_llm.yaml

python scripts/download_tessdata.py         # tessdata_best model into data/tessdata/
python scripts/download_mana_symbols.py     # symbol templates into data/mana_templates/
python scripts/download_type_vocabulary.py  # type line words into data/type_vocabulary.json
python scripts/download_benchmark_images.py # OCR benchmark images into images/benchmark/
python scripts/transcribe_cards.py          # OCR every card in scope into data/processed/ocr_cards.jsonl
python -m banhammer.ocr images/benchmark/<card id>.jpg --debug
python scripts/evaluate_ocr.py              # table + outputs/ocr_eval.csv
pytest
```

`evaluate` reads a Scryfall-shaped card JSON and writes `outputs/reports/<card name>.json` and `.md`. `ingest` skips a source whose raw or processed file is already present; pass `--force` to refresh it. Data processed before stage 1 lacks the `formats` metadata, so run `python -m banhammer.ingest --force` and `python -m banhammer.index` again.

`banhammer.eval` sends every gold card as "Test Card" so the judge cannot recall a famous card by name (`--keep-names` turns this off). A combo piece in `data/eval/gold.jsonl` is a card name or a list of cards that fill the same role, and any of them counts for recall; `synergy_pieces` lists deck partners that cannot be found from card text and are not scored; `ban_reason` says whether a banned card is broken by its text or was banned for its metagame results. The headline is binary, banned or not, on the bans the card text can explain; metagame bans are counted apart (see [ARCHITECTURE.md](ARCHITECTURE.md#evaluation)). The table also gives one row of verdicts per gold group. It also prints one row per config and writes every card's result to `outputs/eval/<config>.json`. `configs/naive.yaml` is the baseline without the format filter; `configs/pipeline.yaml` is version 1; `configs/ablation/` adds one stage 2 technique per file (see [ARCHITECTURE.md](ARCHITECTURE.md#ablation)). The configs with `hybrid: true` build BM25 from `data/processed/cards.jsonl`, so `python -m banhammer.ingest` has to run first on a new machine (`data/` is not in git). `python -m banhammer.index` is only needed if your Pinecone project does not have the index yet. The embedding model in `configs/ingest.yaml` and `configs/pipeline.yaml` has to be the same one.

`--debug` saves the zone overlay and the preprocessed crops to `outputs/debug/<card>/`. Use `--photo` for real photos (perspective correction).

The OCR only transcribes single-faced cards with the 2003 or 2015 frame that are not planeswalkers (`banhammer/ocr/scope.py`, about 94% of the indexed cards); the rest keep their Scryfall text. `transcribe_cards.py` downloads their images into `images/cards/` and takes about two hours with 16 cores; it resumes where it stopped. Ingestion uses a card's transcription when there is one (`source: ocr` in the metadata) and its Scryfall text otherwise, so it also works without running it. `python -m banhammer.ingest --only scryfall_cards --card-text ocr` rebuilds the card records from the transcriptions (and fails if there are none); `--card-text scryfall` rebuilds them with Scryfall text for every card, for example to compare the two indexes. The benchmark in `data/eval/ocr_benchmark.jsonl` lists 150 cards drawn uniformly from that population plus a few per risky feature (2003 frame, legendary, white border, low-resolution scan, ...). `scripts/build_ocr_benchmark.py` made it; do not rebuild it unless you want results that cannot be compared with older runs. The benchmark and evaluation scripts read the Scryfall bulk from `data/raw/oracle-cards.jsonl.gz` (pass `--bulk` for another file). Rules text is compared with Oracle text after removing the most common templating changes, so the error still includes some errata. `data/eval/ocr_gold.jsonl` has 30 of those cards transcribed by hand as printed; the evaluation scores them against both references, and the gap between the two is errata, not reading error.

Zone coordinates are in `configs/zones.yaml` (fractions of the 672x936 card) and the OCR settings in `configs/ocr.yaml`.

## Results

Stage 2 ablation on the 37-card gold set (`data/eval/gold.jsonl`). Not run yet. Fill it in from the table `banhammer.eval` prints.

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
