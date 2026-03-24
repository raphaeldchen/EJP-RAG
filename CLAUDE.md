# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Illinois Legal Research RAG system for querying Illinois criminal law (ILCS statutes, Illinois Supreme Court Rules, CourtListener opinions). Pipeline: scrape → chunk → embed → retrieve.

**Goal:** Exceed the accuracy of general-purpose LLMs (ChatGPT, DeepSeek, Claude) on complex Illinois criminal law queries, with cited sources. Target users are criminal justice researchers.

## Planned Architecture Evolution

The system is being developed toward:

- **Hybrid retrieval:** Supabase vector DB + a graph DB (Neo4j or similar) to capture relationships between statutes, case law, and rules
- **Hosted LLM for query analysis:** Replace local Ollama with a capable hosted model for query decomposition, classification, and rewriting before retrieval — keeping Ollama as a fallback or for embedding only
- **Multi-source opinion corpus:** Harvard Caselaw Access Project (opinions through 2018) + CourtListener API (2018–present) to replace the current stub-heavy opinion dataset

**Design constraints:**
- Accuracy and source attribution take priority over latency
- Scope is Illinois criminal law only (do not expand to other jurisdictions or practice areas without explicit direction)

## Running the Pipeline

```bash
# Full local pipeline (no S3 required)
python ingest/ilga_ingest.py --chapters 720 730 --no-upload --delay 0.75
python chunk/ilga_chunk.py --local-input ilcs_corpus.jsonl        # → chunked_output/ilcs_chunks.jsonl
python embed/ilga_embed.py --local-input chunked_output/ilcs_chunks.jsonl

# Full S3 pipeline (production)
python ingest/ilga_ingest.py --chapters 720 730 --delay 0.75      # uploads to S3
python chunk/ilga_chunk.py                                         # reads/writes S3
python embed/ilga_embed.py                                         # reads S3, writes Supabase

# Query the system (runs built-in test queries)
python -m retrieval.main

# CourtListener opinions (separate pipeline)
python ingest/courtlistener_ingest.py --local-only
python chunk/courtlistener_chunk.py --local-only --limit 1000
```

## Dependencies

No `requirements.txt` exists — infer from imports. Key packages:
- `boto3`, `requests`, `beautifulsoup4` — scraping/S3
- `tiktoken` (cl100k_base) — token counting
- `supabase` — vector DB client
- `rank_bm25` — lexical search
- `sentence-transformers` — CrossEncoder reranking
- `llama-index` — query engine/retriever framework
- `ollama` — local LLM/embedding client

## Environment Variables (`.env`)

```
RAW_S3_BUCKET=illinois-legal-corpus-raw
CHUNKED_S3_BUCKET=illinois-legal-corpus-chunked
COURTLISTENER_S3_PREFIX=courtlistener/
ILCS_S3_PREFIX=ilcs/
SUPREME_COURT_RULES_S3_PREFIX=illinois-supreme-court-rules/
COURTLISTENER_API_TOKEN=...
SUPABASE_URL=...
SUPABASE_SERVICE_KEY=...
OLLAMA_BASE_URL=http://localhost:11434  # default
```

## Architecture

### Pipeline Phases

**1. Ingest** (`ingest/`)
- `ilga_ingest.py` — scrapes ilga.gov; handles two page layouts (Type A: inline text, Type B: TOC with sub-pages); resumes from `.done_acts` checkpoint; outputs `ilcs_corpus.jsonl`
- `courtlistener_ingest.py` — downloads CourtListener bulk CSVs from S3, filters to Illinois courts (ill, illappct), chains dockets → clusters → opinions

**2. Chunk** (`chunk/`)
- `ilga_chunk.py` — character-based splitting with subsection boundaries (`(a)`, `(b)`, etc.); 1500-char chunks, 200-char overlap; outputs `ilcs_chunks.jsonl`
- `courtlistener_chunk.py` — token-aware (tiktoken); detects section headings (Roman numerals, procedural headers); target 600 tokens, max 800, 75-token overlap
- `iscr_chunk.py` — preserves hierarchical structure (article → part → rule → subsection)

**3. Embed** (`embed/`)
- `ilga_embed.py` / `iscr_embed.py` — reads chunks from S3, embeds via Ollama (`nomic-embed-text`, 768-dim), upserts to Supabase in batches of 200; checkpoint-based resumption
- Enriched text prepends structural headers (chapter/act/section) before embedding

**4. Retrieval** (`retrieval/`)
- `config.py` — Supabase URL/key, RPC names, top-k defaults
- `vector_store.py` — custom `SupabaseRPCVectorStore` calling `match_ilcs_chunks` / `match_court_rule_chunks` RPC functions
- `bm25_store.py` — loads all chunks from Supabase on startup, builds in-memory BM25Okapi index; custom tokenizer preserves statute citation patterns
- `indexes.py` — `FusionRetriever` combining vector + BM25 with Reciprocal Rank Fusion (k=60 dampening)
- `postprocessor.py` — CrossEncoder reranking (`ms-marco-MiniLM-L-6-v2`); drops results below score threshold 0.1; returns top-6
- `query_engine.py` — `LLMSingleSelector` router dispatching to ILCS vs ISCR retriever based on query intent
- `reflection.py` — query classification (in_scope / out_of_scope / ambiguous) + rewriting; deterministic rules for common patterns (self-defense, probation, etc.), falls back to LLM
- `main.py` — entry point; builds RAG system and runs test queries

### Retrieval Flow

```
Query → reflection.py (classify + rewrite) → query_engine.py (route ILCS vs ISCR)
     → indexes.py (vector search + BM25 → RRF merge)
     → postprocessor.py (CrossEncoder rerank, threshold filter)
     → Ollama LLM (synthesize answer)
     → Extract citations from source node metadata
```

### Supabase Schema

**`ilcs_chunks`** — chunk_id, parent_id, chunk_index, chunk_total, source, section_citation, chapter_num, act_id, major_topic, text, enriched_text, metadata (JSONB), embedding (pgvector 768-dim)

**`court_rule_chunks`** — chunk_id, source, hierarchical_path, article_number, article_title, part_letter, part_title, rule_number, rule_title, subsection_id, effective_date, amendment_history, text, enriched_text, embedding (pgvector 768-dim)

RPC functions: `match_ilcs_chunks(query_embedding, match_count)`, `match_court_rule_chunks(query_embedding, match_count)`

## Retrieval Evaluation

Test suite lives in `eval/`. Two-step workflow:

```bash
# 1. Generate golden dataset from actual Supabase citations (one-time, ~30s + API cost)
python -m eval.generate_dataset           # → eval/dataset.json (65 cases)

# 2. Run evaluation across pipeline stages
python -m eval.run_eval                            # all 4 stages
python -m eval.run_eval --with-reflection          # add reflection/rewriting as 5th stage
python -m eval.run_eval --scope                    # test out-of-scope rejection accuracy
python -m eval.run_eval --filter-difficulty hard --failures  # drill into worst cases
```

**Target benchmarks** (reranked stage, macro-averaged):

| Metric    | Demo threshold | Refined product |
|-----------|---------------|-----------------|
| Hit@6     | ≥ 0.72        | ≥ 0.88          |
| MRR       | ≥ 0.58        | ≥ 0.73          |
| Recall@6  | ≥ 0.55        | ≥ 0.70          |
| nDCG@6    | ≥ 0.60        | ≥ 0.75          |
| Scope acc.| ≥ 0.90        | ≥ 0.97          |

**Eval validity warning — citation pinning and hardcoded mappings inflate scores:**
- The key citation mappings in `reflection.py` are baked into the same LLM that generates `eval/dataset.json`, so the dataset will over-represent patterns the system already handles perfectly.
- Citation pinning in `indexes.py` bypasses ranking entirely for queries where reflection rewrites to an explicit ILCS citation — those cases score trivially well.
- Always compare "reranked" vs. "reflected" stage scores. A large positive delta means you're measuring the hardcoded lookup table, not retrieval quality.
- TODO: add `--no-pinning` flag to `eval/run_eval.py` to measure true retrieval quality independent of citation pinning.

## Planned Improvements

### High priority (largest accuracy impact)

- [ ] **Domain-adapted embeddings** — swap `nomic-embed-text` for a legal-domain model (e.g. fine-tune on ILCS/ISCR text, or evaluate `BAAI/bge-large-en-v1.5` as a drop-in). Single highest-leverage change; expect +5–8 nDCG points.
- [ ] **Expand reranker candidate window** — increase `top_n` from 6 to 10 in `CrossEncoderReranker`. Improves Recall@6 on hard multi-statute queries by giving more candidates a chance to survive reranking.
- [ ] **Larger/better reranker** — `cross-encoder/ms-marco-electra-base` over `ms-marco-MiniLM-L-6-v2`. ~3x latency cost but meaningful precision gains, especially on legal text the small model wasn't trained on.
- [ ] **Embed CourtListener opinions** — currently only ILCS + ISCR are in the retrieval path. Adding case law is required for questions about judicial interpretation, constitutional challenges, and sentencing precedent.

### Architecture (planned evolution)

- [ ] **Graph DB layer (Neo4j or similar)** — capture statute→case law→rule relationships to support multi-hop retrieval (e.g. "cases that interpreted 720 ILCS 5/7-1").
- [ ] **Harvard Caselaw Access Project** — opinions through 2018 as a bulk corpus supplement to CourtListener.
- [ ] **Query decomposition** — for hard multi-statute queries, decompose into sub-queries, retrieve separately, merge results. Would directly address low Recall@6 on hard cases.

### Eval / measurement

- [ ] **`--no-pinning` flag in `eval/run_eval.py`** — disable citation pinning during eval to isolate true retrieval quality from hardcoded heuristics. The gap between pinned and unpinned scores reveals dependence on the lookup table.
- [x] **Removed hardcoded citation mappings from `reflection.py`** — reflection now relies on the LLM's own knowledge of Illinois law for citation rewriting. Citation pinning in `indexes.py` remains as a safety net. Mappings were brittle, didn't generalize, and inflated eval scores.
- [ ] **Human-verified eval subset** — manually validate ~20 hard cases in `eval/dataset.json` against the actual statute text, since LLM-generated expected citations may have errors on obscure sections.
- [ ] **Baseline comparison script** — run the same eval queries through Claude/ChatGPT without RAG and measure citation accuracy, to quantify the RAG system's improvement over the general-purpose baseline.

## Key Implementation Notes

- The scraper rate-limits at 0.75s/request by default (`--delay` flag)
- `ilcs_corpus.jsonl.done_acts` is the checkpoint file for resuming ILCS scraping
- BM25 index is rebuilt in-memory on every `retrieval/main.py` startup (loads all chunks from Supabase)
- Ollama must be running locally with `nomic-embed-text` pulled for embedding, and `llama3.2` for inference
- CourtListener chunks are not currently embedded into Supabase (only ILCS + ISCR are in the retrieval path)
