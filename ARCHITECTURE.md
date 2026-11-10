# Architecture

Banhammer evaluates a Magic: The Gathering card against the existing card pool and the Comprehensive Rules, and reports wording problems, dangerous synergies and whether the card looks too strong for its cost. The card is audited against one format, Modern or Standard.

The online part is a short LangGraph: five nodes, always in the same order. Flags in the YAML config turn the advanced RAG techniques of stage 2 on and off; with all of them off the graph behaves exactly like version 1. The offline part downloads the catalog and the rules, then embeds them into Pinecone.

## Overview

```
OFFLINE
  Scryfall bulk ───────▶ scryfall_cards ──┐
                                          ├──▶ data/processed/*.jsonl ──▶ indexing ──▶ Pinecone (cards, rules)
  Comprehensive Rules ─▶ comp_rules ──────┘

ONLINE
  card.json ─▶ build_queries ─▶ retrieve ─▶ exclude_self ─▶ rerank ─▶ generate ─▶ BalanceReport
                    │              │
               (decompose)   dense + BM25 → RRF
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
- `configs/ablation/*.yaml` add the stage 2 flags one at a time (see [Ablation](#ablation)). Every flag is optional and off by default:

| Key | Default | Effect |
| --- | --- | --- |
| `decompose` | `false` | The LLM decomposes the card and the queries are expanded (`build_queries`). |
| `hybrid` | `false` | Every catalog query also runs against BM25; dense and lexical lists are fused with RRF. |
| `processed_dir` | `data/processed` | Where BM25 reads `cards.jsonl`. |
| `type_mana_filter` | `false` | Counterpart queries are also filtered by card type and mana value. Only has an effect with `decompose: true`. |
| `rerank` | `none` | `none`, `llm` or `cross_encoder`. |
| `candidates` | `25` | How many fused cards go to the reranker. |
| `context_cards` | no limit | How many cards the judge sees. The rerank configs use `5`. |
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
- **`CardAnalysis`**: the decomposition behind the expanded queries. `abilities` (each one an `Ability` with `text`, `trigger`, `cost`, `effect`, `restrictions`), `counterparts` (each one a `Counterpart`: Oracle-style text of the other half of a combo, with an optional `card_type` and `max_mana_value`) and `rules_topics`. The field descriptions are part of the schema the LLM sees.
- **`BalanceReport`**: wording issues, combos, a power assessment, and a `risk` of `ok`, `watch` or `likely_ban` with a confidence. Every wording issue cites rule hits and every combo cites card hits; see `generate` below.

The LangGraph state holds `card`, `format` (`modern` or `standard`), `analysis` (`CardAnalysis` or `None`), `queries`, `hits`, `report` and `dropped_claims`.

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

`banhammer.pipeline.graph.build_graph` reads a config (`configs/pipeline.yaml` by default) and compiles this graph. The logic of each node lives in module functions (`pipeline/query.py`, `pipeline/retrieval.py`, `pipeline/rerank.py`, `judge` in `graph.py`) so it can be tested without the graph.

1. **`build_queries`**. One query per namespace, both using the whole card text. With `format_filter`, the catalog query carries the filter `formats $in [<format>]` (`card_filter`), so Pinecone only searches cards of the audited format (filtering before retrieval, not after).
   With `decompose: true` the LLM first returns a `CardAnalysis` (one structured call), stored in `analysis`, and the queries are expanded. The two version 1 queries are kept (the catalog one still finds comparable cards). Each ability adds a rules query (its text plus its primitives), and so does each rules topic (up to 5). Each counterpart adds a catalog query (up to 5) with the format filter. A counterpart is the *other half of a combo*, not a similar card: for a card that triggers when an opponent draws, it reads like "Each opponent draws a card." With `type_mana_filter: true` a counterpart query also requires `type_words $in [<type>]` and `cmc $lte <max mana value>`, joined with `$and`.
2. **`retrieve`**. Similarity search in Pinecone, top `k` per query (10 by default). A metadata filter on the query is passed through to Pinecone. With `hybrid: true` every catalog query also runs against an in-memory BM25 index (`rank-bm25`) built once from `data/processed/cards.jsonl`; the same metadata filter is evaluated in Python (`matches`, supporting `$and`, `$in` and `$lte`). Rules are only searched densely.
   The lists of each namespace are fused with **reciprocal rank fusion**: every list adds `1 / (60 + rank)` to each id it contains, duplicates are merged and `source` keeps the union (`dense`, `bm25`, `bm25+dense`). A namespace with a single list is left untouched, so version 1 keeps its hits and scores as Pinecone returned them.
3. **`exclude_self`**. Drops hits whose name is the evaluated card, or one of its faces.
4. **`rerank`**. With `rerank: none` it does nothing. Otherwise the first `candidates` cards (25 by default) are reordered and the remaining cards stay behind them; rules are never reranked.
   - `llm`: one structured call with the candidates labeled `[C1]…[Cn]`, asking for a 0–10 score of how likely each one forms an infinite loop, a lock or a degenerate engine with the card. Missing labels score 0, and ties keep the RRF order.
   - `cross_encoder`: `cross-encoder/ms-marco-MiniLM-L-6-v2` scores the pairs `(card text, candidate text)`. There is no labelled dataset of broken interactions to train one, so a pretrained model is used. It needs the optional extra `pip install -e ".[rerank]"` (it pulls in torch).
5. **`generate`**. The judge sees every rule hit and the first `context_cards` card hits (all of them by default). Numbers the hits `[C1]`, `[R1]`, … (cards banned in a format are flagged) and asks the LLM for a `BalanceReport`. Then the citations are checked: a label that does not match a hit, or a card label cited as a rule, is removed; a wording issue left without a rule hit, or a combo left without a card hit, is discarded. The kept labels are replaced with the real hit ids and the number of discarded claims is stored in `dropped_claims`.

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

- **recall@5, recall@10 and recall@25**: fraction of the known combo pieces among the first 5, 10 or 25 card hits (cards with combo pieces only), measured on the final card order after fusion and reranking. recall@25 is the coverage of the reranker's candidates before the cut to 5. With a single catalog query (`naive`, `pipeline`) only `k` = 10 cards come back, so recall@25 equals recall@10.
- **risk_accuracy**: how often `risk` matches the expected level.
- **false_alarm_rate**: among the cards expected `ok`, how often the report says `watch` or `likely_ban`.
- **dropped_per_card**: claims discarded per card for lacking a valid citation, a measure of ungrounded output.

## Ablation

Each row adds one technique to the previous one, except row 6, which is an alternative to row 5. Run them all with `python -m banhammer.eval --configs configs/naive.yaml configs/pipeline.yaml configs/ablation/*.yaml`.

| # | Config | Adds |
| --- | --- | --- |
| 0 | `naive` | Baseline: whole card text, dense search over the whole catalog |
| 1 | `pipeline` | Format filter before retrieval (version 1) |
| 2 | `ablation/1_decompose` | Decomposition and query expansion toward mechanical counterparts |
| 3 | `ablation/2_hybrid` | BM25 + RRF in the catalog |
| 4 | `ablation/3_type_mana_filter` | Type and mana value filter on the counterpart queries |
| 5 | `ablation/4_rerank_llm` | LLM reranking of 25 candidates, top 5 to the judge |
| 6 | `ablation/5_rerank_cross_encoder` | Cross-encoder instead of the LLM as reranker |

Row 5 changes two things at once, `rerank: llm` and `context_cards: 5`, because the proposal sends the top 5 to the judge. Comparing its recall@25 with row 4 isolates the candidate pool; recall@5 isolates the reranking.

## Package layout

```
src/banhammer/
├── models.py              # CardInput, CardAnalysis, Query, Hit, BalanceReport
├── shared/                # config, card text, JSONL, reports
├── providers/             # OpenAI chat, OpenAI embeddings, Pinecone
├── ingest/                # scryfall_cards, comp_rules
├── indexing/              # JSONL → Pinecone
├── pipeline/
│   ├── state.py
│   ├── query.py           # version 1 queries, decomposition, query expansion
│   ├── retrieval.py       # card_filter, matches, BM25Index, rrf
│   ├── rerank.py          # LLM and cross-encoder rerankers
│   └── graph.py           # the five nodes and the judge
├── eval/
├── ocr/                   # image → text, separate from the graph
└── __main__.py
```
