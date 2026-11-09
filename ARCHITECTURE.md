# Architecture

Banhammer evaluates a Magic: The Gathering card against the existing card pool and the Comprehensive Rules, and reports wording problems, dangerous synergies and whether the card looks too strong for its cost. The card is audited against one format, Modern or Standard.

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
| Card catalog | Scryfall bulk data, `oracle_cards` file. The download URL is read from `https://api.scryfall.com/bulk-data` (field `jsonl_download_uri`, gzipped JSONL, one card per line). | One record per Oracle card that is `legal` or `banned` in Modern or Standard. Non-playable layouts (`art_series`, `token`, `double_faced_token`, `emblem`, `planar`, `scheme`, `vanguard`, `front_card`) are dropped. |
| Comprehensive Rules | Plain-text file linked from `https://magic.wizards.com/en/rules` (the file name changes with every rules update, so the link is scraped from that page). | One record per numbered rule and subrule (`702.19`, `702.19a`, ...), with its `Example:` lines. The table of contents, the glossary and the credits are skipped. |

Scryfall asks for an identifiable `User-Agent`; it is built from `BANHAMMER_CONTACT`.

Banned cards stay in the catalog on purpose: they are the clearest examples of what breaks the format, and the generator sees them flagged as `banned`.

## Configuration and secrets

- `configs/pipeline.yaml` sets the chat model, the embedding model, Pinecone, how many hits to keep (`k`) and whether catalog searches are restricted to the audited format (`format_filter`). `configs/naive.yaml` is the same without the format filter, kept as the evaluation baseline.
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
- **`BalanceReport`**: wording issues, combos, a power assessment, and a `risk` of `ok`, `watch` or `likely_ban` with a confidence. Every wording issue cites rule hits and every combo cites card hits; see `generate` below.

The LangGraph state holds `card`, `format` (`modern` or `standard`), `queries`, `hits`, `report` and `dropped_claims`.

## Embeddings and the index

Chat and embeddings both go through OpenAI (`gpt-4o-mini` and `text-embedding-3-small` in the default config). Vectors live in one Pinecone index. The index name includes the embedding model, for example `banhammer-text-embedding-3-small`, because vectors from different models cannot be compared. Switching models means building a new index.

## Offline: ingestion and indexing

### Ingestion (`python -m banhammer.ingest`)

Each source has `download()` and `normalize()`. Raw files go to `data/raw/` and normalized records to `data/processed/<namespace>.jsonl`. A source is skipped when its file is already there.

- **`scryfall_cards`**: text is `"<name> | <type_line> | <mana_cost> | <oracle_text>"`. Faces of double-faced, split and adventure cards are joined. Metadata: `oracle_id`, `name`, `mana_cost`, `cmc`, `colors`, `types` (type line before the dash), `type_words` (the same as a list, e.g. `["Legendary", "Creature"]`), `keywords`, `formats` (the formats where the card is legal or banned), `banned` (the formats where it is banned) and the text.
- **`comp_rules`**: text is the rule number and its text, followed by its examples. Metadata: `rule`, `parent`, `title`, `section` and the text. Rules are split on the rule number at the start of each line.

Cards and rules are short, so each record is one document. Nothing is chunked further.

### Indexing (`python -m banhammer.index`)

Reads the JSONL files, embeds the text and upserts it into one Pinecone index with two namespaces:

| Namespace | Records | Searched for |
| --- | --- | --- |
| `cards` | One per card legal or banned in Modern or Standard | Synergies, combo pieces and comparable cards |
| `rules` | One per rule and subrule | Rules that govern the card's abilities and wording |

The record id is the `oracle_id` for cards and the rule number for rules, so re-indexing overwrites instead of duplicating. Indexing never downloads anything. It only reads `data/processed/`.

## Online: the evaluation pipeline

`banhammer.pipeline.graph.build_graph` reads `configs/pipeline.yaml` and compiles this graph:

1. **`build_queries`**. One query per namespace, both using the whole card text. With `format_filter`, the catalog query carries the filter `formats $in [<format>]`, so Pinecone only searches cards of the audited format (filtering before retrieval, not after).
2. **`retrieve`**. Similarity search in Pinecone, top `k` per query (10 by default). A metadata filter on the query is passed through to Pinecone.
3. **`exclude_self`**. Drops hits whose name is the evaluated card, or one of its faces.
4. **`generate`**. Numbers the hits `[C1]`, `[R1]`, … (cards banned in a format are flagged) and asks the LLM for a `BalanceReport`. Then the citations are checked: a label that does not match a hit, or a card label cited as a rule, is removed; a wording issue left without a rule hit, or a combo left without a card hit, is discarded. The kept labels are replaced with the real hit ids and the number of discarded claims is stored in `dropped_claims`.

`python -m banhammer evaluate <card.json> --format modern|standard` runs the graph and writes `outputs/reports/<card name>.json` and `.md`.

## Evaluation

`python -m banhammer.eval` runs a gold set (`data/eval/gold.jsonl`) through one or more pipeline configs, prints a comparison table and writes each card's result to `outputs/eval/<config>.json`. The evaluated card is excluded by `exclude_self`.

The gold set has 37 cards, each with the format it is audited against, its known combo pieces (if any) and the expected `risk`:

| Group | Format | Expected | Cards | Source |
| --- | --- | --- | --- | --- |
| Banned combo and engine cards | Modern | `likely_ban` | 10 | Modern banned list |
| Legal two- and three-card combos | Modern | `watch` | 9 | Commander Spellbook (`source` holds the variant id) |
| Ordinary staples | Modern | `ok` | 10 | Hand-picked |
| Banned cards | Standard | `likely_ban` | 4 | Scryfall `banned:standard` |
| Ordinary cards | Standard | `ok` | 4 | Scryfall `f:standard` |

Metrics:

- **recall@5 and recall@10**: fraction of the known combo pieces among the first 5 or 10 card hits (cards with combo pieces only).
- **risk_accuracy**: how often `risk` matches the expected level.
- **false_alarm_rate**: among the cards expected `ok`, how often the report says `watch` or `likely_ban`.
- **dropped_per_card**: claims discarded per card for lacking a valid citation, a measure of ungrounded output.

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
