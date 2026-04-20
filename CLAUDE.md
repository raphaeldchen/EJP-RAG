# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Illinois Legal Research RAG system for querying Illinois criminal law. Pipeline: scrape → chunk → embed → retrieve.

**Goal:** Exceed the accuracy of general-purpose LLMs (ChatGPT, DeepSeek, Claude) on complex Illinois criminal law queries, with cited sources. Target users are criminal justice researchers.

**Corpus sources (as of 2026-04-19):**

| Source | Script | Output | Contents |
|--------|--------|--------|----------|
| ILCS statutes | `ilga_ingest.py` | `ilcs_corpus.jsonl` | Illinois Compiled Statutes (ch. 720, 730, …) |
| IL Supreme Court Rules | `iscr_chunk.py` | direct to chunks | Illinois Supreme Court Rules |
| IL court opinions | `cap_static_download.py` | `cap_bulk_corpus.jsonl` | IL Supreme + Appellate opinions 1819–2024 (CAP bulk) |
| 7th Circuit opinions | `courtlistener_ingest.py` + `courtlistener_api.py` | `cl_filtered/` → chunked | Federal 7th Circuit opinions (CourtListener supplement) |
| IL Admin Code | `iac_ingest.py` | `iac_corpus.jsonl` | IDOC-relevant Title 20 regulations (519 sections) |
| IDOC directives | `idoc_ingest.py` | `idoc_corpus.jsonl` | Administrative directives + reentry resources (103 records) |
| SPAC publications | `spac_ingest.py` | `spac_corpus.jsonl` | Sentencing Policy Advisory Council reports (166 records) |
| ICCB reports | `iccb_ingest.py` | `iccb_corpus.jsonl` | Correctional education enrollment reports FY2020–2025 |
| Federal docs | `federal_ingest.py` | `federal_corpus.jsonl` | Federal Register rules, BOP policy, ED guidance |
| Restore Justice | `restorejustice_ingest.py` | `restorejustice_corpus.jsonl` | Advocacy org resources (HTML + PDFs) |
| Cook County PD | `cookcounty_pd_ingest.py` | `cookcounty_pd_corpus.jsonl` | Public Defender resources |

## Week of 2026-04-15 — Current Focus

Three priorities in order:

1. **Ingest court opinions** — CAP bulk download in progress (`cap_static_download.py --after 1980-01-01`); supplement with CourtListener 7th Circuit after; then embed into retrieval path
2. **Chunking validation** — audit ILCS and ISCR chunking with a spot-check test suite before any re-embedding work
3. **Embedding model decision** — evaluate `intfloat/e5-large-v2`; decide whether to fine-tune `nomic-embed-text` or switch models

## Planned Architecture Evolution

The system is being developed toward:

- **Hybrid retrieval:** Supabase vector DB + a graph DB (Neo4j or similar) to capture relationships between statutes, case law, and rules
- **Hosted LLM for query analysis:** Replace local Ollama with a capable hosted model for query decomposition, classification, and rewriting before retrieval — keeping Ollama as a fallback or for embedding only

**Design constraints:**
- Accuracy and source attribution take priority over latency
- Scope is Illinois criminal law only (do not expand to other jurisdictions or practice areas without explicit direction)

## Running the Pipeline

All ingest scripts use `--local-only` to skip S3 upload. All use `python3`.

```bash
# ILCS statutes
python3 ingest/ilga_ingest.py --chapters 720 730 --delay 0.75 --local-only
python3 chunk/ilga_chunk.py --local-input ilcs_corpus.jsonl
python3 embed/ilga_embed.py --local-input chunked_output/ilcs_chunks.jsonl

# IL Supreme Court Rules
python3 chunk/iscr_chunk.py --local-only
python3 embed/iscr_embed.py --local-only

# Court opinions — see dedicated section below

# IL Admin Code (IDOC Title 20)
python3 ingest/iac_ingest.py --local-only

# IDOC directives + reentry
python3 ingest/idoc_ingest.py --local-only

# SPAC publications
python3 ingest/spac_ingest.py --local-only

# ICCB correctional education reports
python3 ingest/iccb_ingest.py --local-only

# Federal documents
python3 ingest/federal_ingest.py --local-only

# Restore Justice
python3 ingest/restorejustice_ingest.py --local-only

# Cook County PD
python3 ingest/cookcounty_pd_ingest.py --local-only

# Query the system (runs built-in test queries)
python3 -m retrieval.main
```

## Court Opinions Ingestion Pipeline

Court opinions come from two sources with non-overlapping coverage:

| Source | Coverage | Script |
|--------|----------|--------|
| CAP (Harvard Caselaw Access Project) | IL state courts, 1819–2024 | `cap_static_download.py` |
| CourtListener | 7th Circuit (federal) only | `courtlistener_ingest.py` + `courtlistener_api.py` |

CAP is the primary source. CourtListener supplements only with 7th Circuit federal opinions — Illinois state court opinions in CourtListener overlap heavily with CAP (CourtListener originally ingested from CAP), so they are excluded to avoid duplicate embeddings.

### Step 1 — CAP Bulk Download

Data lives at `https://static.case.law` organized by reporter series and volume. Five Illinois reporters are downloaded:

| Reporter | Court | Volumes | Era |
|----------|-------|---------|-----|
| `ill` | IL Supreme Court (1st series) | ~334 | 1819–1955 |
| `ill-2d` | IL Supreme Court (2nd series) | ~242 | 1955–2011 |
| `ill-app` | IL Appellate (1st series) | ~310 | historical |
| `ill-app-2d` | IL Appellate (2nd series) | ~133 | |
| `ill-app-3d` | IL Appellate (3rd series) | ~297 | |

```bash
# Full run (several hours — checkpoints every case, safe to interrupt/resume)
python3 ingest/cap_static_download.py --after 1980-01-01 --local-only

# Upload to S3 when done
aws s3 cp cap_bulk_corpus.jsonl s3://illinois-legal-corpus-raw/cap/cap_bulk_corpus.jsonl
```

**How it works:** For each volume, fetches `CasesMetadata.json` (lightweight — contains `decision_date` and `file_name` for each case). Date filter applied at this stage — cases outside the range are skipped without downloading. Qualifying cases are fetched individually from `cases/{file_name}.json`. Output schema matches `cap_bulk_ingest.py` (source = `"cap_bulk"`).

**Checkpoint:** `cap_bulk_corpus.jsonl.done` stores written case IDs. Re-running the same command resumes from where it left off.

**Test run:**
```bash
python3 ingest/cap_static_download.py --reporters ill-2d --limit 50 --after 1980-01-01 --local-only
```

### Step 2 — CourtListener 7th Circuit Supplement

Run after the CAP download. Fetches 7th Circuit federal opinions not covered by CAP.

```bash
# 2a. Download and filter CourtListener bulk CSVs (dockets, clusters, opinions for ca7)
python3 ingest/courtlistener_ingest.py --local-only
# → cl_filtered/dockets.csv, clusters.csv, opinions.csv

# 2b. Fetch full opinion text from CourtListener API + chunk inline
#     (requires COURTLISTENER_API_TOKEN in .env)
python3 ingest/courtlistener_api.py --local-only
# → chunked_output/api_opinion_chunks.jsonl
```

**How it works:** `courtlistener_ingest.py` downloads CourtListener's public bulk CSVs from their S3 bucket (`com-courtlistener-storage`), filters to `ca7` only, and uploads filtered CSVs to your S3. `courtlistener_api.py` reads the filtered opinion IDs from S3, fetches full text per opinion from the CL REST API, and chunks inline (token-aware, section-heading detection) — bypassing the normal chunk pipeline. Output lands directly in `chunked_output/`.

**Note:** `courtlistener_api.py` does chunking inline rather than outputting raw JSONL. This is intentional — CL opinions have variable structure requiring specialized section detection. The output `api_opinion_chunks.jsonl` is ready for embedding directly.

### Step 3 — Embed Court Opinions

(Pending — chunker for CAP bulk output not yet written.)

```bash
# TODO: chunk cap_bulk_corpus.jsonl → chunked_output/cap_opinion_chunks.jsonl
# Then embed both:
python3 embed/opinion_embed.py --local-input chunked_output/cap_opinion_chunks.jsonl
python3 embed/opinion_embed.py --local-input chunked_output/api_opinion_chunks.jsonl
```

## Dependencies

No `requirements.txt` exists — infer from imports. Key packages:
- `boto3`, `requests`, `beautifulsoup4` — scraping/S3
- `pypdf` — PDF text extraction (IDOC, SPAC, ICCB, federal, Restore Justice)
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
- `cap_static_download.py` — downloads IL court opinions from `static.case.law` by reporter/volume; uses `CasesMetadata.json` for date pre-filtering; checkpoints per case; outputs `cap_bulk_corpus.jsonl`
- `courtlistener_ingest.py` — downloads CourtListener bulk CSVs, filters to `ca7` (7th Circuit) only, uploads filtered dockets/clusters/opinions CSVs to S3
- `courtlistener_api.py` — reads filtered opinion IDs from S3, fetches full text from CL REST API, chunks inline; outputs `chunked_output/api_opinion_chunks.jsonl`
- `iac_ingest.py` — scrapes IDOC Title 20 Illinois Admin Code from `ilga.gov/agencies/JCAR/EntirePart`; 15 IDOC-relevant parts, 519 sections
- `idoc_ingest.py` — downloads IDOC Administrative Directives (PDF) via AEM JSON manifest; scrapes reentry resources page
- `spac_ingest.py` — enumerates ICJIA file server directory, downloads SPAC publication PDFs; de-duplicates timestamp-suffixed variants
- `iccb_ingest.py` — downloads ICCB Annual Enrollment & Completion reports (PDF) FY2020–2025
- `federal_ingest.py` — Federal Register API, Congress.gov HTML, BOP PDF, ED Dear Colleague Letters
- `restorejustice_ingest.py` — crawls Restore Justice IL website; handles both HTML sub-pages and PDFs
- `cookcounty_pd_ingest.py` — Cook County Public Defender resources

**2. Chunk** (`chunk/`)
- `ilga_chunk.py` — character-based splitting with subsection boundaries (`(a)`, `(b)`, etc.); 1500-char chunks, 200-char overlap; outputs `ilcs_chunks.jsonl`
- `courtlistener_chunk.py` — token-aware (tiktoken); detects section headings (Roman numerals, procedural headers); target 600 tokens, max 800, 75-token overlap
- `iscr_chunk.py` — preserves hierarchical structure (article → part → rule → subsection)
- CAP bulk chunker — not yet written; needed before opinions can be embedded

**3. Embed** (`embed/`)
- `ilga_embed.py` / `iscr_embed.py` — reads chunks from S3, embeds via Ollama (`nomic-embed-text`, 768-dim), upserts to Supabase in batches of 200; checkpoint-based resumption
- Enriched text prepends structural headers (chapter/act/section) before embedding

**4. Retrieval** (`retrieval/`)
- `config.py` — Supabase URL/key, RPC names, top-k defaults
- `vector_store.py` — custom `SupabaseRPCVectorStore` calling `match_ilcs_chunks` / `match_court_rule_chunks` RPC functions
- `bm25_store.py` — loads all chunks from Supabase on startup, builds in-memory BM25Okapi index; custom tokenizer preserves statute citation patterns
- `indexes.py` — `FusionRetriever` combining vector + BM25 with Reciprocal Rank Fusion (k=60 dampening); `DualFusionRetriever` runs both ILCS and ISCR sub-retrievers in parallel and merges via RRF (replaces `RouterQueryEngine`)
- `postprocessor.py` — CrossEncoder reranking (`ms-marco-MiniLM-L-6-v2`); drops results below score threshold -3.0; returns top-6
- `query_engine.py` — `RetrieverQueryEngine` wrapping `DualFusionRetriever`; no routing — both corpora always searched
- `reflection.py` — query classification (in_scope / out_of_scope / ambiguous) + rewriting; LLM-based only (hardcoded citation mappings removed)
- `main.py` — entry point; builds RAG system and runs test queries

### Retrieval Flow

```
Query → reflection.py (classify + rewrite) → indexes.py (DualFusionRetriever)
     → [ILCS: vector + BM25 → RRF] + [ISCR: vector + BM25 → RRF] → merge via RRF
     → postprocessor.py (CrossEncoder rerank, threshold filter)
     → Claude LLM (synthesize answer)
     → Extract citations from source node metadata
```

**Why `DualFusionRetriever` instead of a router:** The old `LLMSingleSelector` router would pick either ILCS or ISCR per query. Cross-domain queries (answer spans both corpora) scored empty results. Running both in parallel and merging via RRF fixes this by construction — hard cross-domain R@6 improved from 0.409 → 0.515. BM25 is intentionally kept corpus-agnostic: relevant chunks that appear in both sub-retrievers' BM25 results receive double RRF votes, amplifying the signal for cross-domain queries.

### Supabase Schema

**`ilcs_chunks`** — chunk_id, parent_id, chunk_index, chunk_total, source, section_citation, chapter_num, act_id, major_topic, text, enriched_text, metadata (JSONB), embedding (pgvector 768-dim)

**`court_rule_chunks`** — chunk_id, source, hierarchical_path, article_number, article_title, part_letter, part_title, rule_number, rule_title, subsection_id, effective_date, amendment_history, text, enriched_text, embedding (pgvector 768-dim)

RPC functions: `match_ilcs_chunks(query_embedding, match_count)`, `match_court_rule_chunks(query_embedding, match_count)`

## Retrieval Evaluation

Test suite lives in `eval/`. Two-step workflow:

```bash
# 1. Generate golden dataset from actual Supabase citations (one-time, ~30s + API cost)
python3 -m eval.generate_dataset           # → eval/dataset.json (65 cases)

# 2. Run evaluation across pipeline stages
python3 -m eval.run_eval                            # all 4 stages
python3 -m eval.run_eval --with-reflection          # add reflection/rewriting as 5th stage
python3 -m eval.run_eval --scope                    # test out-of-scope rejection accuracy
python3 -m eval.run_eval --filter-difficulty hard --failures  # drill into worst cases
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

### Model improvement priority order

Leverage ranking for this corpus (highest to lowest):

1. **Embedding model** — affects recall at the source; if the right chunk isn't in the top-40 candidate pool, the reranker never sees it. The adjacent-section failure (e.g. 104-10 vs 104-25) is an embedding failure. Fixing it helps every query.
2. **Reranker** — affects precision on near-misses that survive into the candidate pool. Valuable but downstream of embeddings.
3. **Prompt engineering** — cheaper than fine-tuning; worth trying before generation fine-tuning.
4. **Generation model (Claude)** — lowest ROI. Anthropic does not expose fine-tuning on the Claude API. More importantly, generation is not the bottleneck — retrieval is. Claude already synthesizes well from good context; improving what gets retrieved matters more.

### High priority (largest accuracy impact)

- [ ] **Write CAP bulk chunker** — `cap_bulk_corpus.jsonl` needs a chunker before opinions can be embedded. Can reuse `courtlistener_chunk.py` logic (token-aware, section-heading detection) with minor schema adaptation for the `cap_bulk` source field.
- [ ] **Embed court opinions** — currently only ILCS + ISCR are in the retrieval path. Adding case law (CAP + CourtListener 7th Circuit) is required for questions about judicial interpretation, constitutional challenges, and sentencing precedent.
- [ ] **Fine-tune embedding model** — `nomic-embed-text` is the best out-of-the-box model tested so far and remains the production baseline. Alternatives tested and ruled out: `BAAI/bge-base-en-v1.5` (nDCG@6 0.317, -0.28 vs nomic), `mxbai-embed-large` (nDCG@6 0.389, -0.21 vs nomic). Neither closed the gap without fine-tuning. Next step: fine-tune `nomic-embed-text` or `BAAI/bge-large-en-v1.5` on lawyer-verified (query, relevant chunk) pairs using contrastive/bi-encoder loss (MultipleNegativesRankingLoss in sentence-transformers). Hard negatives already available from eval failures. **Highest single-leverage change. Planned after lawyer refinement sessions.** Expect +5–8 nDCG points.

  **Remaining candidate to evaluate before committing to fine-tuning:**
  - `intfloat/e5-large-v2` (sentence-transformers, 1024-dim — requires prepending `"query: "`/`"passage: "` prefixes at embed time; designed specifically for retrieval)
  - Skip `bge-large` (Ollama-native) — mxbai already tested the 1024-dim Ollama path; bge-large is unlikely to beat nomic given the pattern observed
  - Skip legal-specific BERT models (`pile-of-law`, `InLegalBERT`) — trained for classification not dense retrieval

  **Embed infrastructure:** `EMBED_BACKEND`, `EMBED_MODEL`, `EMBED_DIM`, `ILCS_TABLE`, `ILCS_RPC`, `ISCR_TABLE`, `ISCR_RPC` are all env-var configurable in `config.py`. Experiment tables use naming convention `ilcs_chunks_{model}` / `court_rule_chunks_{model}` with matching RPC functions. See `embed/ilga_embed.py --table` and `embed/iscr_embed.py --table` flags.

  **Schema note:** 1024-dim models require `ALTER TABLE ... TYPE vector(1024)` on experiment tables. mxbai required `MAX_EMBED_CHARS=1500` in `iscr_embed.py` (BERT 512-token limit; legal text tokenizes at ~3 chars/token).

  **Fine-tuning does not transfer between models** — weights are architecture-specific. Training data (query/chunk pairs) is reusable across any model.
- [ ] **Fine-tune reranker on Illinois legal query pairs** — `cross-encoder/ms-marco-electra-base` tested and significantly worse (nDCG@6 0.484 vs 0.579); reverted. The right path is fine-tuning MiniLM (`ms-marco-MiniLM-L-6-v2`) on domain-specific triplets: (query, correct chunk, hard-negative near-miss chunk). Hard negatives already available from eval failures. Needs lawyer-verified positives as ground truth — 100–200 pairs sufficient. **Planned with lawyer collaboration.**
- [ ] **Expand reranker candidate window** — increase `top_n` from 6 to 10 in `CrossEncoderReranker`. Improves Recall@6 on hard multi-statute queries by giving more candidates a chance to survive reranking.

### Chunking (needs evaluation)

**Before investing further in embedding or reranker tuning, audit whether the right content is present and intact in the chunks** — malformed chunks upstream invalidate retrieval improvements downstream.

**~~Fixed~~ — ISCR semantic unit severing (2026-04-20)**
Rule 401(a) and 1,168 other ISCR rule_subsection chunks were starting with bare numeric markers `(1)`, `(2)`, `(3)` with no parent clause context. Fixed in `iscr_chunk.py` by splitting only at letter-subsection boundaries `(a)(b)(c)` — numeric items now stay within their parent letter subsection. Also fixed: 403 chunks had `[PAGE N]` markers bleeding into chunk text from `merge_pages_to_text`; stripped in `chunk_document` with a single regex. Test suite in `tests/test_iscr_chunk.py` covers both fixes.

**~~Diagnosis complete~~ — 705 ILCS 405/5-915 not a chunking issue**
Chapter 705 (Juvenile Court Act) was never ingested — `ilga_ingest.py` was only run with `--chapters 720 730`. The section does not exist in `ilcs_corpus.jsonl`. Fix: run `python3 ingest/ilga_ingest.py --chapters 705 --delay 0.75 --local-only`. The retrieval instability (q028 vs. q064) cannot be diagnosed until the corpus is populated. Test `test_5_915_chunk_is_self_contained` in `tests/test_ilga_chunk.py` skips automatically until then.

**Known issues in `ilga_chunk.py` — found by test suite 2026-04-20**

Four bugs confirmed by `python3 -m pytest tests/test_ilga_chunk.py` (4 failing, 1 skipped):

1. **Orphaned subsection starts (2,569 chunks)** — `test_no_orphaned_subsection_starts` fails. Chunks at `chunk_index > 0` start with bare `(a)`, `(b)`, `(c)` etc. with no parent clause. `split_on_subsections` intentionally cuts at these boundaries, so each subsection becomes its own chunk. Unlike ISCR (where chunks must be self-contained), ILCS chunks get structural metadata prepended as `enriched_text` before embedding. **Before fixing:** verify whether the orphaned starts actually hurt retrieval by running the eval with and without the fix. May need the same letter-only split approach applied to `ilga_chunk.py`.

2. **Oversized chunks (12 chunks exceed `CHUNK_SIZE=1500`)** — `test_no_chunk_exceeds_max_size` fails. Subsections that contain no sentence-ending punctuation never trigger `split_by_sentences`, producing single chunks larger than the target. Worst case: 2,735 chars. Fix: add a hard character-limit fallback split (e.g. split at whitespace) after `split_by_sentences` when the segment still exceeds `CHUNK_SIZE`.

3. **Undersized chunk (1 chunk below `MIN_CHUNK_SIZE`)** — `test_no_empty_chunks` fails. One chunk: `ch725-act1977-sec150_S17_c0`. Likely a very short section that passed the `MIN_CHUNK_SIZE=100` filter but shouldn't have. Low priority.

4. **Overlap not preserved across sentence splits (42 pairs)** — `test_sentence_split_overlap` fails. The `split_by_sentences` function uses a 200-char overlap buffer, but the buffer tracks characters not sentences — the last sentence of chunk N often exceeds 200 chars and is not carried into chunk N+1 at all. Fix: in `split_by_sentences`, after building the overlap buffer, always include at least the last complete sentence of the previous chunk regardless of character count.

**Diagnostic approach (before rechunking)**
1. Pull all chunks for a known-failing section directly from Supabase (filter by `section_citation`) and inspect split points
2. Check whether the right chunk appears in the pre-reranking candidate pool (40 candidates) for failing queries — absent = embedding/BM25 problem; present but dropped = reranker problem; incomplete text = chunking problem
3. Run eval before and after any chunker change to confirm retrieval improvement

- [ ] **Fix ILCS orphaned subsection starts** — investigate whether the 2,569 orphaned-start chunks hurt eval scores before deciding on a fix. If confirmed harmful, apply letter-only split logic (same as ISCR fix) to `split_on_subsections` in `ilga_chunk.py`.
- [ ] **Fix ILCS oversized chunks** — add hard character-limit fallback in `chunk_section` after `split_by_sentences` when a segment still exceeds `CHUNK_SIZE`.
- [ ] **Fix ILCS overlap logic** — update `split_by_sentences` to always carry at least the last complete sentence of chunk N into chunk N+1, regardless of the 200-char overlap budget.
- [ ] **Ingest chapter 705** — run `python3 ingest/ilga_ingest.py --chapters 705 --delay 0.75 --local-only` to add the Juvenile Court Act; then re-run `test_5_915_chunk_is_self_contained`.
- [ ] **Write chunk spot-check script** — query Supabase by `section_citation` / `rule_number`, print chunks in order with text previews; use to diagnose known failures before deciding whether rechunking is required.
- [ ] **Distinguish chunking vs. retrieval failures** — for each hard eval failure, determine whether the issue is (a) content not in any chunk, (b) content in a chunk but not surfacing in the top-40 candidate pool, or (c) content in the candidate pool but dropped by the reranker. These require different fixes.

### Architecture (planned evolution)

- [ ] **Graph DB layer (Neo4j or similar)** — capture statute→case law→rule relationships to support multi-hop retrieval (e.g. "cases that interpreted 720 ILCS 5/7-1").
- [ ] **Query decomposition** — for hard multi-statute queries, decompose into sub-queries, retrieve separately, merge results. Would directly address low Recall@6 on hard cases.

### Eval / measurement

- [ ] **`--no-pinning` flag in `eval/run_eval.py`** — disable citation pinning during eval to isolate true retrieval quality from hardcoded heuristics. The gap between pinned and unpinned scores reveals dependence on the lookup table.
- [x] **Removed hardcoded citation mappings from `reflection.py`** — reflection now relies on the LLM's own knowledge of Illinois law for citation rewriting. Citation pinning in `indexes.py` remains as a safety net. Mappings were brittle, didn't generalize, and inflated eval scores.
- [ ] **Human-verified eval subset** — manually validate ~20 hard cases in `eval/dataset.json` against the actual statute text, since LLM-generated expected citations may have errors on obscure sections.
- [ ] **Baseline comparison script** — run the same eval queries through Claude/ChatGPT without RAG and measure citation accuracy, to quantify the RAG system's improvement over the general-purpose baseline.

## Key Implementation Notes

- All ingest scripts use `--local-only` flag (consistent across all scripts); omit it to also upload to S3
- Use `python3` (not `python`) on this system
- The scraper rate-limits at 0.75s/request by default (`--delay` flag on `ilga_ingest.py`)
- `ilcs_corpus.jsonl.done_acts` is the checkpoint file for resuming ILCS scraping
- `cap_bulk_corpus.jsonl.done` is the checkpoint file for resuming CAP bulk download (case-ID granularity)
- BM25 index is rebuilt in-memory on every `retrieval/main.py` startup (loads all chunks from Supabase)
- Ollama must be running locally with `nomic-embed-text` pulled for embedding, and `llama3.2` for inference
- Court opinions (CAP + CourtListener) are not yet embedded into Supabase — chunker for CAP bulk output is the blocking step
