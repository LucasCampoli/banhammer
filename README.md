# Banhammer

Balance auditor for Magic: The Gathering cards. Give it a new card and a format (Modern or Standard), and it uses RAG over the Scryfall catalog and the Comprehensive Rules to judge whether the card is `ok`, worth to `watch`, or a `likely_ban`. The report explains why, and every claim cites a card or rule it retrieved.

```
card.json ─▶ build queries ─▶ retrieve (Pinecone) ─▶ drop the card itself ─▶ rerank ─▶ LLM judge ─▶ report
```

Claims that do not cite a retrieved card or rule are dropped. An optional OCR stage reads cards from images. [ARCHITECTURE.md](ARCHITECTURE.md) explains how the pieces fit together.

## Setup

Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"         # add ,rerank for the cross-encoder reranker (pulls in torch)
```

Copy [.env.example](.env.example) to `.env` and set `BANHAMMER_CONTACT` (sent to Scryfall in the User-Agent), `OPENAI_API_KEY` and `PINECONE_API_KEY`.

Tesseract is only needed for OCR, and pip does not install it:

- Linux: `sudo apt install tesseract-ocr`
- macOS: `brew install tesseract`
- Windows: installer from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki), then add it to `PATH` or set `TESSERACT_CMD` to the `.exe`.

## 1. Prepare the data

Run once on a new machine (`data/` is not in git), and again to refresh the data.

```bash
python -m banhammer.ingest      # Scryfall bulk + Comprehensive Rules -> data/processed/
python -m banhammer.index       # embed data/processed/ into Pinecone (cards, rules)
```

- `ingest` skips a source that is already downloaded; `--force` refreshes it.
- `index` is only needed if your Pinecone project does not have the index yet. The embedding model in `configs/ingest.yaml` and in the pipeline configs must be the same.
- Optional: `python -m banhammer.ingest.ban_announcements` reads Wizards' banned and restricted announcements and writes why each card was banned to `data/processed/ban_reasons.jsonl`, for configs with `ban_reasons: true`.

## 2. Audit a card

From the command line, with a Scryfall-shaped card JSON:

```bash
python -m banhammer evaluate path/to/card.json --format modern    # or standard
```

It writes `outputs/reports/<card name>.json` and `.md`. `--config` picks another pipeline, for example `configs/ablation/4_rerank_llm.yaml`.

From the browser:

```bash
python -m banhammer.web         # then open http://127.0.0.1:8000
```

The web UI lets you forge a card or start from an existing one, judge it, and read the report with the text and image of every card and rule it cites.

## 3. Evaluate the pipeline

The gold set `data/eval/gold.jsonl` has 41 real cards with a known verdict. `data/eval/pairs.jsonl` has 9 pairs of invented cards, one broken and one fair version of each, that the model cannot know from memory.

```bash
python -m banhammer.eval --configs configs/no_rag.yaml configs/naive.yaml configs/pipeline.yaml
python -m banhammer.eval --configs configs/naive.yaml --gold data/eval/pairs.jsonl
python -m banhammer.eval --configs configs/ablation/7_decompose_gpt5_mini.yaml --retrieval-only   # recall only, no judge
python -m banhammer.eval --rescore outputs/eval/naive.json   # score a saved run again, no API calls
python -m banhammer.eval.memorization                        # how many gold cards the model recognizes
```

- Every card is sent as "Test Card" so the judge cannot answer from the name (`--keep-names` turns this off).
- The table prints one row per config, and every card's result goes to `outputs/eval/<config>.json`.
- The main metric is balanced accuracy on banned or not banned. Responding always the same scores 0.5.

What each metric means and how configs are compared is in [docs/plan-evaluacion.md](docs/plan-evaluacion.md). What we learned is in [docs/aprendizajes.md](docs/aprendizajes.md).

### Configs

Each YAML in `configs/` is one pipeline variant.

| Config | What it is |
| --- | --- |
| `no_rag` | The judge reads only the card, no retrieval. The baseline. |
| `naive` | One dense search with the card text, over the whole catalog. |
| `pipeline` | `naive` plus a format filter: only cards legal or banned in the audited format. Version 1, and the default. |
| `ablation/*` | `pipeline` plus one advanced RAG technique per file: decomposition, hybrid BM25 search, type and mana filters, LLM or cross-encoder rerank, split context. |
| `no_rag_free`, `no_rag_rule`, `no_rag_catalog` | Judges without retrieval that may use what the model knows. |
| `naive_hybrid`, `naive_reasons` | `naive` with a judge that may also name cards it knows, or that reads why each banned card was banned. |
| `oracle_pieces` | `naive` plus the right combo pieces from the gold set, as if retrieval were perfect. |

Configs with `hybrid: true` build BM25 from `data/processed/cards.jsonl`, so they need `ingest` first.

## 4. OCR

The OCR reads a card image and transcribes it. It does not feed the judge directly: it transcribes the catalog, and `ingest` indexes those transcriptions.

```bash
python scripts/download_tessdata.py          # tessdata_best model -> data/tessdata/
python scripts/download_mana_symbols.py      # symbol templates -> data/mana_templates/
python scripts/download_type_vocabulary.py   # type line words -> data/type_vocabulary.json

python -m banhammer.ocr path/to/card.jpg --debug    # read one card; --photo for real photos
python scripts/transcribe_cards.py                  # transcribe every card in scope -> data/processed/ocr_cards.jsonl
python -m banhammer.ingest --only scryfall_cards --card-text ocr   # rebuild the cards from the transcriptions

python scripts/download_benchmark_images.py  # OCR benchmark images -> images/benchmark/
python scripts/evaluate_ocr.py               # score the OCR -> outputs/ocr_eval.csv
```

- Scope: single-faced cards with the 2003 or 2015 frame that are not planeswalkers, about 94% of the indexed cards (`src/banhammer/ocr/scope.py`). The rest keep their Scryfall text.
- `transcribe_cards.py` downloads the images to `images/cards/` and takes about two hours with 16 cores; it resumes where it stopped. Ingestion works without it, using Scryfall text.
- `--debug` saves the zone overlay and the preprocessed crops to `outputs/debug/<card>/`.
- Zone coordinates are in `configs/zones.yaml` and the OCR settings in `configs/ocr.yaml`.
- The benchmark (`data/eval/ocr_benchmark.jsonl`, 150 cards) and the hand-transcribed `data/eval/ocr_gold.jsonl` are fixed; rebuilding the benchmark with `scripts/build_ocr_benchmark.py` makes results incomparable with older runs.

## Tests

```bash
pytest
```

## Repository map

| Path | What is there |
| --- | --- |
| `src/banhammer/ingest/` | Download and normalize the Scryfall catalog, the rules and the ban announcements |
| `src/banhammer/index.py`, `indexing/` | Embed the processed data into Pinecone |
| `src/banhammer/pipeline/` | The LangGraph: queries, retrieval, rerank and judge |
| `src/banhammer/web/` | Local web UI |
| `src/banhammer/eval/` | Evaluation, metrics and memorization probe |
| `src/banhammer/ocr/` | Card OCR |
| `configs/` | Pipeline, ingest and OCR settings |
| `data/eval/` | Gold set, invented pairs and OCR benchmark (versioned) |
| `docs/` | Evaluation plan, learnings and run results (in Spanish) |

## Notes

Card images and card data belong to Wizards of the Coast, so `data/raw`, `data/processed`, `images/` and `outputs/` are gitignored. Only `data/eval/`, which is hand-written, is versioned. Parts of the code were written with Claude Code.
