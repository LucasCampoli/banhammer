# Architecture

Banhammer evaluates a Magic: The Gathering card against the existing card pool and the Comprehensive Rules, and reports wording problems, dangerous synergies and whether the card looks too strong for its cost. The target format is Modern.

The online part is a short LangGraph: four nodes, always in the same order. The offline part downloads the catalog and the rules, then embeds them into Pinecone.

## Overview

```
OFFLINE
  Scryfall bulk ───────▶ scryfall_cards ──┐
                                          ├──▶ data/processed/*.jsonl ──▶ indexing ──▶ Pinecone (cards, rules)
  Comprehensive Rules ─▶ comp_rules ──────┘

ONLINE
  card.json ─▶ build_queries ─▶ retrieve ─▶ exclude_self ─▶ generate ─▶ BalanceReport
```

OCR is a separate command. It reads a card image and prints the text. It does not feed the graph.

The offline part runs once per data refresh. The online part runs once per evaluated card.

## Data sources

| Source | Where it comes from | What is kept |
| --- | --- | --- |
| Card catalog | Scryfall bulk data, `oracle_cards` file. The download URL is read from `https://api.scryfall.com/bulk-data` (field `jsonl_download_uri`, gzipped JSONL, one card per line). | One record per Oracle card that is `legal` or `banned` in Modern. Non-playable layouts (`art_series`, `token`, `double_faced_token`, `emblem`, `planar`, `scheme`, `vanguard`, `front_card`) are dropped. |
| Comprehensive Rules | Plain-text file linked from `https://magic.wizards.com/en/rules` (the file name changes with every rules update, so the link is scraped from that page). | One record per numbered rule and subrule (`702.19`, `702.19a`, ...), with its `Example:` lines. The table of contents, the glossary and the credits are skipped. |

Scryfall asks for an identifiable `User-Agent`; it is built from `BANHAMMER_CONTACT`.

Banned cards stay in the catalog on purpose: they are the clearest examples of what breaks the format, and the generator sees them flagged as `banned`.

## Configuration and secrets

- `configs/pipeline.yaml` sets the chat model, the embedding model, Pinecone, and how many hits to keep (`k`).
- `configs/ingest.yaml` sets the data sources and the index settings.
- The embedding model in both files has to be the same one.
- Secrets live in `.env` (never committed):

| Variable | Used by |
| --- | --- |
| `OPENAI_API_KEY` | The chat model and the embeddings |
| `PINECONE_API_KEY` | The vector index |
| `BANHAMMER_CONTACT` | Scryfall `User-Agent` |
| `TESSERACT_CMD` | Optional path to the Tesseract binary |

## Data contracts

Pydantic models in `banhammer/models.py`:

- **`CardInput`**: the card to evaluate. A subset of the Scryfall card object: `name`, `mana_cost`, `type_line`, `oracle_text`, `power`, `toughness`, `loyalty`, `card_faces`. `cmc`, `colors` and `keywords` are derived when missing.
- **`Query`**: a text to search, the namespace it targets (`cards` or `rules`), and an optional metadata filter.
- **`Hit`**: a retrieved document: id, namespace, text, metadata, score and `source`.
- **`BalanceReport`**: wording issues, combos, a power assessment, and a `risk` of `ok`, `watch` or `likely_ban` with a confidence. A wording issue with no rule ids, or a combo with no card ids, is marked as speculation.

The LangGraph state holds `card`, `queries`, `hits` and `report`.

## Embeddings and the index

Chat and embeddings both go through OpenAI (`gpt-4o-mini` and `text-embedding-3-small` in the default config). Vectors live in one Pinecone index. The index name includes the embedding model, for example `banhammer-text-embedding-3-small`, because vectors from different models cannot be compared. Switching models means building a new index.

## Offline: ingestion and indexing

### Ingestion (`python -m banhammer.ingest`)

Each source has `download()` and `normalize()`. Raw files go to `data/raw/` and normalized records to `data/processed/<namespace>.jsonl`. A source is skipped when its file is already there.

- **`scryfall_cards`**: text is `"<name> | <type_line> | <mana_cost> | <oracle_text>"`. Faces of double-faced, split and adventure cards are joined. Metadata: `oracle_id`, `name`, `mana_cost`, `cmc`, `colors`, `types`, `keywords`, `modern` (`legal` or `banned`) and the text.
- **`comp_rules`**: text is the rule number and its text, followed by its examples. Metadata: `rule`, `parent`, `title`, `section` and the text. Rules are split on the rule number at the start of each line.

Cards and rules are short, so each record is one document. Nothing is chunked further.

### Indexing (`python -m banhammer.index`)

Reads the JSONL files, embeds the text and upserts it into one Pinecone index with two namespaces:

| Namespace | Records | Searched for |
| --- | --- | --- |
| `cards` | One per Modern-legal or Modern-banned card | Synergies, combo pieces and comparable cards |
| `rules` | One per rule and subrule | Rules that govern the card's abilities and wording |

The record id is the `oracle_id` for cards and the rule number for rules, so re-indexing overwrites instead of duplicating. Indexing never downloads anything. It only reads `data/processed/`.

## Online: the evaluation pipeline

`banhammer.pipeline.graph.build_graph` reads `configs/pipeline.yaml` and compiles this graph:

1. **`build_queries`**. One query per namespace, both using the whole card text.
2. **`retrieve`**. Similarity search in Pinecone, top `k` per query (10 by default). A metadata filter on the query is passed through to Pinecone.
3. **`exclude_self`**. Drops hits whose name is the evaluated card, or one of its faces.
4. **`generate`**. Numbers the hits `[C1]`, `[R1]`, … and asks the LLM for a `BalanceReport`. The labels in the answer are replaced with the real hit ids.

`python -m banhammer evaluate <card.json>` runs the graph and writes `outputs/reports/<card name>.json` and `.md`.

## Evaluation

`python -m banhammer.eval` runs a gold set (`data/eval/gold.jsonl`) through one or more pipeline configs and prints a comparison table. The gold set holds cards banned in Modern with their known combo pieces, plus ordinary cards that should not raise alarms. The evaluated card is excluded by `exclude_self`.

- **Retrieval**: recall@10 of the known combo pieces.
- **Generation**: how often `risk` matches the expected level.

## Package layout

```
src/banhammer/
├── models.py              # CardInput, Query, Hit, BalanceReport
├── shared/                # config, card text, JSONL, reports
├── providers/             # OpenAI chat, OpenAI embeddings, Pinecone
├── ingest/                # scryfall_cards, comp_rules
├── indexing/              # JSONL → Pinecone
├── pipeline/
│   ├── state.py
│   └── graph.py           # the four nodes
├── eval/
├── ocr/                   # image → text, separate from the graph
└── __main__.py
```
