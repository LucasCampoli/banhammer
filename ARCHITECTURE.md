# Architecture

Banhammer evaluates a Magic: The Gathering card against the existing card pool and the Comprehensive Rules, and reports wording problems, dangerous synergies and whether the card looks too strong for its cost. The target format is Modern.

It is a retrieval-augmented generation (RAG) pipeline built with LangGraph and LangChain. Every stage is a module that can be switched on or off from a YAML file.

## Overview

```
OFFLINE
  Scryfall bulk ───────▶ scryfall_cards ──┐
                                          ├──▶ data/processed/*.jsonl ──▶ indexing ──▶ Pinecone (cards, rules)
  Comprehensive Rules ─▶ comp_rules ──────┘                                                 │
                                                                                            │
ONLINE                                                                                      ▼
  card.json (or image ─▶ OCR) ─▶ CardInput ─▶ build_query ─▶ [expand_query] ─▶ retrieve ─▶ [fusion]
                                                                                            │
                               BalanceReport ◀─ generate ◀─ [rerank] ◀─ [filter] ◀──────────┘
```

Stages in brackets are optional. The offline part runs once per data refresh; the online part runs once per evaluated card.

## Data sources

| Source | Where it comes from | What is kept |
| --- | --- | --- |
| Card catalog | Scryfall bulk data, `oracle_cards` file. The download URL is read from `https://api.scryfall.com/bulk-data` (field `jsonl_download_uri`, gzipped JSONL, one card per line). | One record per Oracle card that is `legal` or `banned` in Modern. Non-playable layouts (`art_series`, `token`, `double_faced_token`, `emblem`, `planar`, `scheme`, `vanguard`, `front_card`) are dropped. |
| Comprehensive Rules | Plain-text file linked from `https://magic.wizards.com/en/rules` (the file name changes with every rules update, so the link is scraped from that page). | One record per numbered rule and subrule (`702.19`, `702.19a`, ...), with its `Example:` lines. The table of contents, the glossary and the credits are skipped. |

Scryfall asks for an identifiable `User-Agent`; it is built from `BANHAMMER_CONTACT`.

Banned cards stay in the catalog on purpose: they are the clearest examples of what breaks the format, and the generator sees them flagged as `banned`.

## Configuration and secrets

- `configs/pipeline.yaml` chooses the modules of the online pipeline.
- `configs/ingest.yaml` chooses the data sources and the index settings.
- Secrets live in `.env` (never committed):

| Variable | Used by |
| --- | --- |
| `OPENAI_API_KEY` | `llm` and `embeddings` with provider `openai` |
| `PINECONE_API_KEY` | `vector_store` with provider `pinecone` (indexes are created and queried in Pinecone's hosted service) |
| `HUGGINGFACEHUB_API_TOKEN` | Optional, only for the `huggingface` providers used in local testing |
| `BANHAMMER_CONTACT` | Scryfall `User-Agent` |

## Data contracts

Pydantic models in `banhammer/models.py`, shared by every stage:

- **`CardInput`**: the card to evaluate. A subset of the Scryfall card object: `name`, `mana_cost`, `type_line`, `oracle_text`, `power`, `toughness`, `loyalty`, `card_faces`. `cmc`, `colors` and `keywords` are derived when missing. Any real card can be copied from Scryfall as-is; an invented card is written by hand in the same shape. The OCR stage produces this same model.
- **`Query`**: a text to search plus the namespace it targets (`cards` or `rules`) and an optional metadata filter.
- **`Hit`**: a retrieved document: id, namespace, text, metadata, score and the module that found it.
- **`BalanceReport`**: the output (see [Generation](#generation)).

The LangGraph state is a `TypedDict` holding `card`, `queries`, `hits` and `report`.

## Providers

Factories that build LangChain objects from the YAML, so the rest of the code never imports a vendor package directly.

| Factory | Providers | Notes |
| --- | --- | --- |
| `llm` | `openai` (`ChatOpenAI`), `huggingface` (`ChatHuggingFace`), `fake` | `openai` is the default. `huggingface` is for local testing. `fake` returns fixed answers for unit tests. |
| `embeddings` | `openai` (`OpenAIEmbeddings`, `text-embedding-3-small`), `huggingface` (`HuggingFaceEmbeddings`) | |
| `vector_store` | `pinecone` (`PineconeVectorStore`), `memory` (`InMemoryVectorStore`) | `memory` is for unit tests. |

Vectors from different embedding models cannot be compared, so the Pinecone index name includes the embedding model (for example `banhammer-text-embedding-3-small`). Switching models means building a new index, never mixing both.

## Offline: ingestion and indexing

### Ingestion (`python -m banhammer.ingest`)

Each source is a module with `download()` and `normalize()`. Raw files go to `data/raw/` and normalized records to `data/processed/<namespace>.jsonl`, as LangChain `Document`s (text + metadata). Sources are skipped when their file is already there.

- **`scryfall_cards`**:
  - Text: `"<name> | <type_line> | <mana_cost> | <oracle_text>"`. For double-faced, split and adventure cards, the faces are joined.
  - Metadata: `oracle_id`, `name`, `mana_cost`, `cmc`, `colors`, `types` (the part of `type_line` before the dash), `keywords` (as provided by Scryfall), `modern` (`legal` / `banned`) and the text itself.
- **`comp_rules`**:
  - Text: the rule number and its text, for example `"702.19b The controller of an attacking creature with trample ..."`, followed by its examples.
  - Metadata: `rule` (`702.19b`), `parent` (`702.19`), `title` (`Trample`), `section` (`702. Keyword Abilities`) and the text.
  - Split with a regular expression on the rule number at the start of each line.

Cards and rules are short, so each record is one document; nothing is chunked further.

### Indexing (`python -m banhammer.index`)

Reads the JSONL files, embeds the text and upserts it into **one Pinecone index with two namespaces**:

| Namespace | Records | Searched for |
| --- | --- | --- |
| `cards` | One per Modern-legal or Modern-banned card | Synergies, combo pieces and comparable cards |
| `rules` | One per rule and subrule | Rules that govern the card's abilities and wording |

The record id is the `oracle_id` for cards and the rule number for rules, so re-indexing overwrites instead of duplicating. Indexing never downloads anything; it only reads `data/processed/`.

## Online: the evaluation pipeline

`banhammer.pipeline.graph_builder` reads `configs/pipeline.yaml` and builds a LangGraph `StateGraph` with only the enabled nodes, connected in the order below. Each node type is a small interface with one method, and a registry maps module names to implementations. Removing a module is removing its line from the YAML.

| Node | Interface | Required |
| --- | --- | --- |
| `build_query` | `build(card) -> list[Query]` | yes |
| `expand_query` | `expand(card, queries) -> list[Query]` | no, any number |
| `retrieve` | `search(query, k) -> list[Hit]` | yes, one or more |
| `fusion` | `fuse(hit_lists) -> list[Hit]` | only with more than one retriever |
| `filter` | `apply(card, hits) -> list[Hit]` | no, any number |
| `rerank` | `rerank(card, hits) -> list[Hit]` | no |
| `generate` | `generate(card, hits) -> BalanceReport` | yes |

### Query building

- **`full_text`** (naive): one query per namespace with the whole card text.
- **`per_ability`**: splits `oracle_text` by line (in Oracle text each ability is on its own line) and issues one query per ability. Each line is classified with simple rules: a cost before a colon is an activated ability, a line starting with "When", "Whenever" or "At" is a triggered ability, a keyword is a keyword ability, and anything else is a static ability.

### Query expansion

Adds queries to the list; it never searches by itself.

- **`keyword_rules`**: for each keyword on the card, adds a query to the `rules` namespace filtered by `title` (for example `title = "Trample"`), so the exact rule comes back even when a similarity search would miss it.
- **`multi_query`**: asks the LLM to rewrite each query in a few different ways (LangChain `MultiQueryRetriever`).
- **`synergy_tags`**: tags the card with what it does (`untap`, `copy`, `token_maker`, `tap_ability`, `etb`, `sacrifice`, ...) using regular expressions, and adds queries for the complementary tags from a fixed pair table (`untap` ↔ `tap_ability`, `copy` ↔ `etb`, ...). It needs the `mechanic_tags` step at ingestion, which stores the same tags in the card metadata.

### Retrieval

- **`dense`** (naive): similarity search in Pinecone with the query embedding, top-k per query and namespace. Honors the query's metadata filter.
- **`bm25`**: keyword search over the same `data/processed/*.jsonl`, built in memory (LangChain `BM25Retriever`). Catches exact wording that embeddings blur.
- **`tag_match`**: fetches cards whose `tags` metadata contains the tags asked for by `synergy_tags`.

### Fusion

- **`rrf`**: reciprocal rank fusion of the lists returned by several retrievers (LangChain `EnsembleRetriever`).

### Filters

- **`exclude_self`** (naive): drops the evaluated card from the results, so a real card being re-evaluated does not find itself.
- **`dedupe`**: drops repeated hits coming from different queries.

### Reranking

- **`cross_encoder`**: rescores each (card, hit) pair with a cross-encoder model and keeps the top-k (LangChain `CrossEncoderReranker`).
- **`llm_rerank`**: asks the LLM to order the hits by relevance. Alternative to `cross_encoder`.

### Generation

- **`llm_judge`**: builds a prompt with the card and the numbered hits, and asks the LLM for a `BalanceReport` through structured output (`with_structured_output`). `BalanceReport` contains:
  - `wording_issues`: problems with how the card is written, each citing the rule hits it relies on.
  - `combos`: dangerous interactions, each with the card hits involved and the sequence of plays.
  - `power_assessment`: power versus cost compared with the comparable cards retrieved.
  - `risk`: `ok`, `watch` or `likely_ban`, plus a confidence value.
  - Every claim must cite hit ids; a claim with no citation is marked as speculation.

### Output

`python -m banhammer evaluate <card.json>` runs the graph and writes `outputs/reports/<card name>.json` (the `BalanceReport`) and `.md` (the same report for reading).

## Evaluation

`python -m banhammer.eval` runs a gold set (`data/eval/gold.jsonl`) through one or more pipeline configurations and prints a comparison table. The gold set holds cards banned in Modern with their known combo pieces, plus ordinary cards that should not raise alarms. The evaluated card is always excluded from the index.

- **Retrieval**: recall@10 of the known combo pieces.
- **Generation**: how often `risk` matches the expected level.

Every advanced module is enabled only after it beats the naive configuration on this table.

## Modules by phase

**Phase 1, naive RAG** (a working app end to end):

| Area | Modules |
| --- | --- |
| Core | `models`, `config` + `registry`, `graph_builder`, `cli`, `eval` |
| Providers | `llm` (`openai`, `huggingface`, `fake`), `embeddings` (`openai`, `huggingface`), `vector_store` (`pinecone`, `memory`) |
| Offline | `scryfall_cards`, `comp_rules`, Pinecone indexing |
| Pipeline | `full_text` → `dense` → `exclude_self` → `llm_judge` |

**Phase 2, advanced RAG** (one module at a time, each measured against phase 1):

| Stage | Modules |
| --- | --- |
| Query building | `per_ability` |
| Query expansion | `keyword_rules`, `multi_query`, `synergy_tags` (with `mechanic_tags` at ingestion) |
| Retrieval and fusion | `bm25`, `tag_match`, `rrf` |
| Filters | `dedupe` |
| Reranking | `cross_encoder`, `llm_rerank` |

`keyword_rules`, `synergy_tags`, `mechanic_tags` and `tag_match` are only built if the evaluation shows the problem they solve (keyword rules or combo pieces missing from the results).

**Phase 3, OCR input**: `ocr_input` turns a card image into `CardInput` using the existing `banhammer.ocr` package, as an alternative to writing the JSON by hand.

## Package layout

```
src/banhammer/
├── models.py              # CardInput, Query, Hit, BalanceReport, ingestion records
├── config.py              # YAML loading
├── registry.py            # module name → implementation
├── providers/             # llm, embeddings, vector_store factories
├── ingest/                # scryfall_cards, comp_rules
├── indexing/              # JSONL → Pinecone
├── pipeline/
│   ├── state.py
│   ├── graph_builder.py
│   ├── query/             # full_text, per_ability
│   ├── expansion/         # keyword_rules, multi_query, synergy_tags
│   ├── retrieval/         # dense, bm25, tag_match
│   ├── fusion.py          # rrf
│   ├── filters.py         # exclude_self, dedupe
│   ├── rerank/            # cross_encoder, llm_rerank
│   └── generation/        # llm_judge
├── eval/
├── ocr/                   # existing image → OCRCard stage
└── __main__.py            # CLI
```

The empty `parsing/`, `retrieval/` and `verdict/` packages from the original layout are replaced by `pipeline/query/`, `pipeline/retrieval/` and `pipeline/generation/`.
