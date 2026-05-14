# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Illinois criminal justice RAG system focused on higher education in prison and reentry. Pipeline: scrape → chunk → embed → retrieve.

**Goal:** Exceed the accuracy of general-purpose LLMs (ChatGPT, DeepSeek, Claude) on Illinois criminal justice queries — including correctional education, reentry, sentencing policy, and the federal law intersecting with Illinois prisoners — with cited sources. Target users are researchers, practitioners (public defenders, legal advocates), and incarcerated persons.

**Corpus sources (as of 2026-05-02):**

| Source | Script | Output | Contents |
|--------|--------|--------|----------|
| ILCS statutes | `ilga_ingest.py` | `data_files/corpus/ilcs_corpus.jsonl` | Illinois Compiled Statutes (ch. 720, 730, …) |
| IL Supreme Court Rules | `iscr_chunk.py` | direct to chunks | Illinois Supreme Court Rules |
| IL court opinions | `cap_ingest.py` | `data_files/corpus/cap_bulk_corpus.jsonl` | IL Supreme + Appellate opinions 1819–2024 (CAP bulk) |
| 7th Circuit opinions | `courtlistener_ingest.py` + `courtlistener_api.py` | `data_files/cl_filtered/` → `data_files/chunked_output/` | Federal 7th Circuit opinions (CourtListener supplement) |
| IL Admin Code | `iac_ingest.py` | `data_files/corpus/iac_corpus.jsonl` | IDOC-relevant Title 20 regulations (519 sections) |
| IDOC directives | `idoc_ingest.py` | `data_files/corpus/idoc_corpus.jsonl` | Administrative directives + reentry resources (103 records) |
| SPAC publications | `spac_ingest.py` | `data_files/corpus/spac_corpus.jsonl` | Sentencing Policy Advisory Council reports (166 records) |
| ICCB reports | `iccb_ingest.py` | `data_files/corpus/iccb_corpus.jsonl` | Correctional education enrollment reports FY2020–2025 |
| Federal docs | `federal_ingest.py` | `data_files/corpus/federal_corpus.jsonl` | Federal Register rules, BOP policy, ED guidance |
| Restore Justice | `restorejustice_ingest.py` | `data_files/corpus/restorejustice_corpus.jsonl` | Advocacy org resources (HTML + PDFs) |
| Cook County PD | `cookcounty_pd_ingest.py` | `data_files/corpus/cookcounty_pd_corpus.jsonl` | Public Defender resources |

## Summer 2026 Focus

**Lawyer collaboration begins summer 2026.** The primary focus is retrieval quality — lawyers will use the audit dashboard (`audit_app.py`) to label chunks as BINDING / RELEVANT / IRRELEVANT, generating ground-truth feedback that will drive embedding fine-tuning and reranker training. No persona-specific features until retrieval quality is validated.

1. **Embedding** — all sources now embedded in Supabase (ILCS, ISCR, CAP, IAC, IDOC, SPAC, ICCB, Federal, Restore Justice, Cook County PD)
2. **Embedding model decision** — evaluate `intfloat/e5-large-v2`; decide whether to fine-tune `nomic-embed-text` or switch models
3. **Retrieval quality labeling** — lawyers label pre-rerank and post-rerank candidates in the audit dashboard; feedback stored in `audit_feedback` Supabase table for analysis

**Audit app is live at `https://ejp-rag-audit.com` as of 2026-05-13.** See `docs/deployment/audit-app-deployment.md` for full infrastructure details, SSH access, and known OCI gotchas.

**To deploy updates to the audit app:**
```bash
# 1. Local — commit and push as normal
git add <files> && git commit -m "..." && git push

# 2. Server — pull and restart
ssh ubuntu@163.192.97.229
cd legal_rag && git pull
sudo systemctl restart audit-app
```

**WebSocket keepalive (Cloudflare Tunnel)** — Cloudflare drops idle WebSocket connections after ~100 s. Fixed via two layers (2026-05-14):
1. `audit_app.py`: `_bm25_status_banner` always renders `"● Hybrid retrieval active"` once BM25 is ready, guaranteeing a real Streamlit delta every 5 s.
2. `.streamlit/config.toml`: `enableWebsocketCompression = false` — prevents Cloudflare proxy from desynchronizing the deflate context on long-lived connections.
3. `/etc/cloudflared/config.yml` on the server: `originRequest: tcpKeepAlive: 15s` added to each ingress rule. See `docs/deployment/audit-app-deployment.md` for the exact config block.

**BM25 cache on the server** — lives at `~/legal_rag/data_files/bm25_cache/` (~1.2 GB). Survives `systemctl restart`. Only needs to be re-uploaded if the server is reprovisioned:
```bash
ssh ubuntu@163.192.97.229 "mkdir -p ~/legal_rag/data_files/bm25_cache"
scp -r /Users/raphaelchen/Desktop/legal_rag/data_files/bm25_cache ubuntu@163.192.97.229:~/legal_rag/data_files/
```

## Planned Architecture Evolution

The system is being developed toward:

- **Hybrid retrieval:** Supabase vector DB + a graph DB (Neo4j or similar) to capture relationships between statutes, case law, and rules.

  **Graph-native sources (dual placement — vector DB *and* graph DB):** ILCS statutes, IL Admin Code, IDOC directives, court opinions (CAP + CourtListener), IL Supreme Court Rules. These form the authoritative legal hierarchy: statute → regulation → directive, with opinions citing across all levels. CAP bulk JSON includes structured citation lists, making case→statute edges cheap to extract. Statute cross-references require a regex pass over section text.

  **Vector-only sources:** SPAC, ICCB, federal docs, Restore Justice, Cook County PD. These are analytical/advocacy documents that reference the legal hierarchy but aren't authoritative nodes in it — no meaningful graph traversal starts from or arrives at them.

  **Why dual placement isn't redundant:** The graph answers structural queries ("what cases cited 730 ILCS 5/3-6-3?"); the vector DB answers semantic queries ("find chunks discussing proportionality in sentencing"). They're orthogonal.

  **Retrieval integration:** Do NOT add graph traversal as a third RRF arm — graph-native sources would accumulate an extra RRF vote on every query and crowd out vector-only sources unfairly. Instead, use graph traversal as a **pre-filter / candidate expander**: for structural queries, traverse the graph to get a candidate node set, then run vector search scoped to those nodes. Always run vector + BM25 over all sources in parallel. The CrossEncoder reranker is the single relevance arbiter — it scores purely on content quality regardless of which retrieval arm produced the chunk.
- **Hosted LLM for query analysis:** Replace local Ollama with a capable hosted model for query decomposition, classification, and rewriting before retrieval — keeping Ollama as a fallback or for embedding only

**Design constraints:**
- Accuracy and source attribution take priority over latency
- Scope is Illinois criminal law only (do not expand to other jurisdictions or practice areas without explicit direction)

## Running the Pipeline

All ingest scripts use `--local-only` to skip S3 upload. All use `python3`.

```bash
# ILCS statutes
python3 ingest/ilga_ingest.py --chapters 720 730 --delay 0.75 --local-only
python3 chunk/ilga_chunk.py --local-input data_files/corpus/ilcs_corpus.jsonl
python3 embed/batch_embed.py --source ilcs --local-input data_files/chunked_output/ilcs_chunks.jsonl

# IL Supreme Court Rules
python3 chunk/iscr_chunk.py --local-only
python3 embed/batch_embed.py --source iscr

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
| CAP (Harvard Caselaw Access Project) | IL state courts, 1819–2024 | `cap_ingest.py` |
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
python3 ingest/cap_ingest.py --after 1980-01-01 --local-only

# Upload to S3 when done
aws s3 cp data_files/corpus/cap_bulk_corpus.jsonl s3://illinois-legal-corpus-raw/cap/cap_bulk_corpus.jsonl
```

**How it works:** For each volume, fetches `CasesMetadata.json` (lightweight — contains `decision_date` and `file_name` for each case). Date filter applied at this stage — cases outside the range are skipped without downloading. Qualifying cases are fetched individually from `cases/{file_name}.json`. Output schema matches `cap_ingest.py` (source = `"cap_bulk"`).

**Checkpoint:** `data_files/corpus/cap_bulk_corpus.jsonl.done` stores written case IDs. Re-running the same command resumes from where it left off.

**Test run:**
```bash
python3 ingest/cap_ingest.py --reporters ill-2d --limit 50 --after 1980-01-01 --local-only
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

```bash
python3 embed/opinion_embed.py --local-input data_files/chunked_output/cap_opinion_chunks.jsonl
python3 embed/opinion_embed.py --local-input chunked_output/api_opinion_chunks.jsonl
```

## Virtual Environment

The project uses a `venv/` at the repo root. **Always install packages into the venv, never into the system Python.**

```bash
source venv/bin/activate          # activate before running anything
venv/bin/pip install <package>    # or just `pip install` after activating
```

All `python3` commands in this file assume the venv is active. If you see `ModuleNotFoundError` for a package that should be installed, the venv is likely not activated or the package was installed to the system Python instead.

## Dependencies

`requirements.txt` exists at the repo root (generated from the Mac venv on 2026-05-13, **Python 3.11 required** — `networkx==3.6.1` is incompatible with 3.10). Key packages:
- `boto3`, `requests`, `beautifulsoup4` — scraping/S3
- `pypdf` — PDF text extraction (IDOC, SPAC, ICCB, federal, Restore Justice)
- `tiktoken` (cl100k_base) — token counting
- `supabase` — vector DB client
- `rank_bm25`, `bm25s` — lexical search
- `sentence-transformers` — CrossEncoder reranking
- `llama-index` — query engine/retriever framework
- `ollama` — local LLM/embedding client
- `mcp[cli]` — MCP server (`mcp_server/`)
- `streamlit` — Audit Dashboard (`audit_app.py`)
- `bcrypt` — password hashing for lawyer accounts (`auth/accounts.py`)

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
ANTHROPIC_API_KEY=...
ADMIN_PASSWORD=...          # audit app admin gate
OLLAMA_BASE_URL=http://localhost:11434  # default
```

## Architecture

### Pipeline Phases

**1. Ingest** (`ingest/`)
- `ilga_ingest.py` — scrapes ilga.gov; handles two page layouts (Type A: inline text, Type B: TOC with sub-pages); resumes from `.done_acts` checkpoint; outputs `ilcs_corpus.jsonl`
- `cap_ingest.py` — downloads IL court opinions from `static.case.law` by reporter/volume; uses `CasesMetadata.json` for date pre-filtering; checkpoints per case; outputs `cap_bulk_corpus.jsonl`
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
- `cap_chunk.py` — token-aware (tiktoken); section-heading detection; opinion-type segmentation (majority/dissent/concurrence); originally 1,211,329 chunks from 151,228 opinions; S3 file filtered to 337,505 criminal-relevant chunks (1973+); outputs `cap_opinion_chunks.jsonl`

**3. Embed** (`embed/`)
- `batch_embed.py` — single entry point for all sources; reads chunks from S3 (or `--local-input`), embeds via Ollama (`nomic-embed-text`, 768-dim), upserts to Supabase in batches of 200; `--batch-delay` throttles write pressure; source → table mapping in `SOURCE_REGISTRY`
- Enriched text (with structural headers) is emitted by all chunkers and consumed directly; no per-embed-script header reconstruction

**4. Retrieval** (`retrieval/`)
- `config.py` — Supabase URL/key, RPC names, top-k defaults
- `vector_store.py` — custom `SupabaseRPCVectorStore` calling `match_ilcs_chunks` / `match_court_rule_chunks` RPC functions
- `bm25_store.py` — disk-cached BM25 index (`data_files/bm25_cache/`, ~1.2 GB: 730 MB `corpus.json` + 490 MB numpy index); staleness detected by comparing live Supabase row counts against `meta.json`; loads from cache on startup (~15–45s on server), rebuilds from Supabase only when counts change; loads in a background thread (swapped into `MultiCollectionRetriever` when ready; vector-only fallback until then); custom tokenizer preserves statute citation patterns
- `indexes.py` — `FusionRetriever` (pure vector, per-collection); `MultiCollectionRetriever` runs all collections in parallel plus a single shared BM25 arm, merges all arms via RRF with per-collection weights; `bm25=None` is safe — BM25 arm is silently skipped until the background thread swaps it in
- `postprocessor.py` — CrossEncoder reranking (`ms-marco-MiniLM-L-6-v2`); drops results below score threshold -3.0; returns top-6
- `query_engine.py` — `RetrieverQueryEngine` wrapping `DualFusionRetriever`; no routing — both corpora always searched
- `reflection.py` — query classification (in_scope / out_of_scope / ambiguous) + rewriting; LLM-based only (hardcoded citation mappings removed)
- `main.py` — entry point; builds RAG system and runs test queries

### Retrieval Flow

```
Query → reflection.py (classify + rewrite) → indexes.py (MultiCollectionRetriever)
     → [ilcs: vector] + [iscr: vector] + [opinions: vector] + [regulations: vector]
       + [documents: vector] + [shared BM25 arm] → merge via weighted RRF
     → postprocessor.py (CrossEncoder rerank, threshold filter)
     → Claude LLM (synthesize answer)
     → Extract citations from source node metadata
```

**Why `MultiCollectionRetriever` instead of a router:** The old `LLMSingleSelector` router would pick one corpus per query. Cross-domain queries (answer spans multiple corpora) scored empty results. Running all collections in parallel and merging via RRF fixes this — hard cross-domain R@6 improved from 0.409 → 0.515. BM25 is a single shared arm (one RRF vote regardless of collection count) to avoid inflating its weight as collections are added.

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

- [x] **Write CAP bulk chunker** — `cap_chunk.py` done; original output was 1,211,329 chunks from 151,228 opinions. S3 file has since been filtered to 337,505 criminal-relevant chunks (1973–2011, 1.82 GB) via two passes: date cutoff (`≥ 1973`) then `_is_cap_criminal` (People/In re case name or 705/720/725/730 ILCS citation). Location: `s3://illinois-legal-corpus-chunked/cap/cap_opinion_chunks.jsonl`.
- [x] **Embed court opinions** — CAP + CourtListener 7th Circuit are in the retrieval path. All sources embedded as of 2026-05-11.
- [ ] **Fine-tune embedding model** — `nomic-embed-text` is the best out-of-the-box model tested so far and remains the production baseline. Alternatives tested and ruled out: `BAAI/bge-base-en-v1.5` (nDCG@6 0.317, -0.28 vs nomic), `mxbai-embed-large` (nDCG@6 0.389, -0.21 vs nomic). Neither closed the gap without fine-tuning. Next step: fine-tune `nomic-embed-text` or `BAAI/bge-large-en-v1.5` on lawyer-verified (query, relevant chunk) pairs using contrastive/bi-encoder loss (MultipleNegativesRankingLoss in sentence-transformers). Hard negatives already available from eval failures. **Highest single-leverage change. Planned after lawyer refinement sessions.** Expect +5–8 nDCG points.

  **Remaining candidate to evaluate before committing to fine-tuning:**
  - `intfloat/e5-large-v2` (sentence-transformers, 1024-dim — requires prepending `"query: "`/`"passage: "` prefixes at embed time; designed specifically for retrieval)
  - Skip `bge-large` (Ollama-native) — mxbai already tested the 1024-dim Ollama path; bge-large is unlikely to beat nomic given the pattern observed
  - Skip legal-specific BERT models (`pile-of-law`, `InLegalBERT`) — trained for classification not dense retrieval

  **Embed infrastructure:** `EMBED_BACKEND`, `EMBED_MODEL`, `EMBED_DIM`, `ILCS_TABLE`, `ILCS_RPC`, `ISCR_TABLE`, `ISCR_RPC` are all env-var configurable in `config.py`. Experiment tables use naming convention `ilcs_chunks_{model}` / `court_rule_chunks_{model}` with matching RPC functions. To embed into an experiment table, update the `table` field in the relevant `SOURCE_REGISTRY` entry in `embed/batch_embed.py`.

  **Schema note:** 1024-dim models require `ALTER TABLE ... TYPE vector(1024)` on experiment tables. mxbai required `MAX_EMBED_CHARS=1500` (BERT 512-token limit; legal text tokenizes at ~3 chars/token); `batch_embed.py` uses 2000 chars, safe for nomic-embed-text's 2048-token limit.

  **Fine-tuning does not transfer between models** — weights are architecture-specific. Training data (query/chunk pairs) is reusable across any model.
- [ ] **Fine-tune reranker on Illinois legal query pairs** — `cross-encoder/ms-marco-electra-base` tested and significantly worse (nDCG@6 0.484 vs 0.579); reverted. The right path is fine-tuning MiniLM (`ms-marco-MiniLM-L-6-v2`) on domain-specific triplets: (query, correct chunk, hard-negative near-miss chunk). Hard negatives already available from eval failures. Needs lawyer-verified positives as ground truth — 100–200 pairs sufficient. **Planned with lawyer collaboration.**
- [ ] **Expand reranker candidate window** — increase `top_n` from 6 to 10 in `CrossEncoderReranker`. Improves Recall@6 on hard multi-statute queries by giving more candidates a chance to survive reranking.
- [ ] **Investigate contextual retrieval across all sources** — prepend a brief document-level summary to each chunk's `enriched_text` to give the embedding and reranker richer context. A mid-document paragraph currently has no signal about what the parent document concluded. Two implementation tiers: (1) **rule-based prefix** using existing structured metadata — free, zero hallucination risk, gets opinion type (majority/dissent/concurrence) and citation string from CAP JSON; (2) **LLM-generated summary per document** — one LLM call per case/report (not per chunk), stored in metadata and prepended at embed time. For court opinions the key question is: what was the charge and what did the court hold? For SPAC/policy docs: what policy question does this report address? **Caution:** LLM-generated holding summaries can hallucinate — validate accuracy on 20–30 samples before embedding at scale. **Timing:** best done before any re-embedding run so the improvement is baked in from the start. Start with rule-based (tier 1); assess whether lawyers flag missing holding info during audit sessions before committing to LLM summarization (tier 2).

### Chunking (needs evaluation)

**Before investing further in embedding or reranker tuning, audit whether the right content is present and intact in the chunks** — malformed chunks upstream invalidate retrieval improvements downstream.

**~~Fixed~~ — ISCR semantic unit severing (2026-04-20)**
Rule 401(a) and 1,168 other ISCR rule_subsection chunks were starting with bare numeric markers `(1)`, `(2)`, `(3)` with no parent clause context. Fixed in `iscr_chunk.py` by splitting only at letter-subsection boundaries `(a)(b)(c)` — numeric items now stay within their parent letter subsection. Also fixed: 403 chunks had `[PAGE N]` markers bleeding into chunk text from `merge_pages_to_text`; stripped in `chunk_document` with a single regex. Test suite in `tests/test_iscr_chunk.py` covers both fixes.

**~~Diagnosis complete~~ — 705 ILCS 405/5-915 not a chunking issue**
Chapter 705 (Juvenile Court Act) was never ingested — `ilga_ingest.py` was only run with `--chapters 720 730`. The section does not exist in `ilcs_corpus.jsonl`. Fix: run `python3 ingest/ilga_ingest.py --chapters 705 --delay 0.75 --local-only`. The retrieval instability (q028 vs. q064) cannot be diagnosed until the corpus is populated. Test `test_5_915_chunk_is_self_contained` in `tests/test_ilga_chunk.py` skips automatically until then.

**~~Fixed~~ — ILCS chunking bugs (2026-04-27)**

All four bugs fixed; `python3 -m pytest tests/test_ilga_chunk.py` now passes (16 passed, 1 skipped for uningested ch705):

1. **~~Orphaned subsection starts~~** — Fixed: `_CANDIDATE_SUBSECTION_RE` now splits only on single lowercase letters `([a-z])`, eliminating 1,262 numeric orphans. Added `_pack_subsections` greedy bin-packing to combine adjacent letter subsections that fit within CHUNK_SIZE. Remaining unavoidable orphans (large adjacent subsections) get a section context prefix (full "citation heading" → short `[citation]` → hard-trim).

2. **~~Oversized chunks~~** — Fixed: `split_by_sentences` now hard-splits individual sentences > CHUNK_SIZE inline with overlap, rather than post-hoc. Overlap carry capped to `CHUNK_SIZE - next_sentence_len` to prevent accumulation overflow on large sentences.

3. **~~Undersized chunk~~** — Fixed: added `MIN_CHUNK_SIZE` guard to the single-chunk fast path in `chunk_section`.

4. **~~Overlap not preserved~~** — Fixed: overlap buffer always carries at least the last complete sentence; capped to ensure next sentence fits within CHUNK_SIZE. For hard-split chunks (no terminal punctuation), test uses tail-suffix check instead of full last-sentence match.

**Diagnostic approach (before rechunking)**
1. Pull all chunks for a known-failing section directly from Supabase (filter by `section_citation`) and inspect split points
2. Check whether the right chunk appears in the pre-reranking candidate pool (40 candidates) for failing queries — absent = embedding/BM25 problem; present but dropped = reranker problem; incomplete text = chunking problem
3. Run eval before and after any chunker change to confirm retrieval improvement

- [x] **Fix ILCS orphaned subsection starts** — letter-only split + greedy packing + context prefix. Done 2026-04-27.
- [x] **Fix ILCS oversized chunks** — inline hard-split for oversized sentences + overlap cap. Done 2026-04-27.
- [x] **Fix ILCS overlap logic** — overlap always carries last sentence; capped at CHUNK_SIZE - next_sentence. Done 2026-04-27.
- [ ] **Ingest chapter 705** — run `python3 ingest/ilga_ingest.py --chapters 705 --delay 0.75 --local-only` to add the Juvenile Court Act; then re-run `test_5_915_chunk_is_self_contained`.
- [ ] **Write chunk spot-check script** — query Supabase by `section_citation` / `rule_number`, print chunks in order with text previews; use to diagnose known failures before deciding whether rechunking is required.
- [ ] **Distinguish chunking vs. retrieval failures** — for each hard eval failure, determine whether the issue is (a) content not in any chunk, (b) content in a chunk but not surfacing in the top-40 candidate pool, or (c) content in the candidate pool but dropped by the reranker. These require different fixes.
- [ ] **Fix duplicate chunk_ids in `courtlistener_api.py`** — embedding audit (2026-05-11) found 328 duplicate chunk_ids in `courtlistener/bulk/api_opinion_chunks.jsonl` (13,924 lines → 13,596 unique rows in Supabase after upsert deduplication). The chunker in `courtlistener_api.py` is generating non-unique IDs; fix the chunk_id generation logic (likely needs to incorporate opinion_id + chunk_index rather than whatever it uses now) and re-upload to S3. No re-embedding needed after the fix — just rechunk + re-run `batch_embed --source courtlistener` (checkpoint will skip the 13,596 already present if chunk_ids are stable, or use `--recreate` if IDs change).
- [ ] **Fix SPAC enriched_text header when `category` is empty** (`chunk/spac_chunk.py:269`) — `_make_enriched` uses `rec.get('category', '')` verbatim, so records with no category produce headers like `SPAC  (2012): A Retrospective` (with a blank where the category goes). Fix: skip the category token when empty so the header becomes `SPAC (2012): A Retrospective`. Separately, PDF OCR artifacts like `M A R C H  2 0 1 2` can bleed into section headings — audit how often this occurs before rechunking. **Must fix before lawyer deployment** (lawyers see this text verbatim in the audit dashboard).
- [ ] **Audit OCR noise in PDF-sourced chunks** (SPAC, ICCB, Federal, Restore Justice, IDOC) — spaced-out characters from PDF extraction (e.g. `M A R C H  2 0 1 2`) appear in chunk headers and body text. Determine frequency and severity before deciding whether to add a post-extraction normalization pass in the ingest scripts. **Must assess before lawyer deployment.**

### Architecture (planned evolution)

- [ ] **Graph DB layer (Neo4j or similar)** — capture statute→case law→rule relationships to support multi-hop retrieval (e.g. "cases that interpreted 720 ILCS 5/7-1"). See Planned Architecture Evolution for source placement decisions and retrieval integration design.
- [ ] **Query decomposition** — for hard multi-statute queries, decompose into sub-queries, retrieve separately, merge results. Would directly address low Recall@6 on hard cases.
- [ ] **Differential collection weighting in RRF** — `MultiCollectionRetriever` currently merges all 5 collections with equal RRF weights. Future exploration: weight authoritative sources (statutes, rules, opinions, regulations) higher than advisory documents (SPAC, ICCB, Restore Justice, Cook County PD) to bias the candidate pool before reranking. Run eval with and without to measure impact — the CrossEncoder may already compensate sufficiently.
- [ ] **Persona-specific retrieval strategies** — deferred until after summer 2026 lawyer collaboration. Future work: different retrieval weights, query rewriting styles, or reranker thresholds per audience (researcher / practitioner / incarcerated person). The `audit_feedback` table has a `persona` column reserved for this. Collect retrieval-quality signal first; don't add persona complexity until the baseline retrieval is validated.

### Architecture (code quality — deferred)

Identified 2026-04-30. In priority order:

- [ ] **Shared Chunk interface across all chunkers** (`chunk/`) — 12 chunkers emit incompatible schemas (different field names, `metadata` dict vs flat, hierarchical IDs vs UUIDs). `vector_store.py` and `bm25_store.py` hard-code field names to compensate. Fix: define a shared `Chunk` dataclass with universal fields (`chunk_id`, `text`, `source`, `token_count`, `metadata: dict`) and adapt all chunkers to emit it. *In progress — see current grilling session.*
- [ ] **`_secondary_query` private-state seam** (`retrieval/indexes.py` + `retrieval/main.py`) — caller sets `dual_retriever._secondary_query` before `engine.query()` and clears it in `finally`. Should be a parameter to `retrieve()` instead. Eliminates the fragile mutation + cleanup pattern.
- [ ] **Extract system prompts from source files** (`retrieval/reflection.py`, `retrieval/query_engine.py`) — Query Analysis system prompt (65 lines) and QA answer prompt (26 lines) are embedded in code. Hard to version, compare, or test. Move to `config/prompts/` as plain text or YAML.
- [ ] **Split `postprocessor.py` responsibilities** — currently owns RRF fusion (imported at runtime by `indexes.py`), a second RRF variant, Citation labeling, and cross-encoder reranking. RRF belongs with ranking logic, Citation labeling belongs with answer assembly, reranking is a separate concern. Runtime import is a circular-dependency risk.
- [x] **`BM25Retriever` hardcodes ILCS + ISCR tables** (`retrieval/bm25_store.py`) — resolved by `CollectionConfig` registry in multi-collection retrieval design (2026-05-03).
- [ ] **BM25 nodes reach the LLM without `enriched_text`** (`retrieval/bm25_store.py`, `retrieval/indexes.py`) — BM25 correctly scores on plain `text` (to avoid header inflation), but returns nodes with plain `text` as their body. After RRF fusion, the LLM sees a mix: vector-retrieved Chunks have `enriched_text` (with citation headers), BM25-retrieved Chunks have plain `text` (no headers). Fix: after RRF merge, replace each BM25 node's text with its `enriched_text` before passing to the reranker and LLM. BM25 scoring stays on plain `text`; LLM generation always sees `enriched_text`.

### Eval / measurement

- [ ] **`--no-pinning` flag in `eval/run_eval.py`** — disable citation pinning during eval to isolate true retrieval quality from hardcoded heuristics. The gap between pinned and unpinned scores reveals dependence on the lookup table.
- [x] **Removed hardcoded citation mappings from `reflection.py`** — reflection now relies on the LLM's own knowledge of Illinois law for citation rewriting. Citation pinning in `indexes.py` remains as a safety net. Mappings were brittle, didn't generalize, and inflated eval scores.
- [ ] **Human-verified eval subset** — manually validate ~20 hard cases in `eval/dataset.json` against the actual statute text, since LLM-generated expected citations may have errors on obscure sections.
- [ ] **Baseline comparison script** — run the same eval queries through Claude/ChatGPT without RAG and measure citation accuracy, to quantify the RAG system's improvement over the general-purpose baseline.

## Testing Standards

Every chunking script must have a test suite in `tests/` with two layers:
1. **Unit + corpus tests** — run `chunk_section()` / `chunk_document()` in memory against the raw S3 corpus via a `*_chunks` fixture; cover orphaned subsection starts, contiguous indices, accurate chunk totals, no empty chunks, unique IDs, no oversized chunks, required metadata fields, and any source-specific invariants (normalization, enriched text hierarchy, etc.)
2. **S3 output verification** — a separate `*_chunks_s3` fixture that reads the actual chunked JSONL from the chunked S3 bucket; must include: record count matches in-memory output, no corrupt/missing keys, no empty text, correct source field. This catches write-path bugs (wrong key, truncation, encoding) that in-memory tests cannot.

Test suites exist for: ILCS, ISCR, IAC, ICCB, IDOC, SPAC, Federal, Restore Justice, Cook County PD. Missing: `courtlistener_chunk.py` and `merge_opinion_chunks.py` — these need test suites before the shared Chunk interface migration reaches them. **Known issue:** `courtlistener_api.py` produces 328 duplicate chunk_ids in the current S3 output (confirmed 2026-05-11); the test suite for this script must include a unique-ID assertion.

## Key Implementation Notes

- **Prefer cloud storage over local files** — the canonical home for all data is AWS S3 (raw corpus) and Supabase (chunked/embedded). Local `data_files/` copies are temporary working files only. When writing new ingest or chunk scripts, default to reading from and writing to S3/Supabase, not local disk. Use `--local-only` flags only for development/testing.
- **Check S3 and Supabase first when looking for data** — always check S3 (`aws s3 ls s3://illinois-legal-corpus-raw/` and `aws s3 ls s3://illinois-legal-corpus-chunked/`) and Supabase before concluding a file or dataset doesn't exist. Local `data_files/` may be absent even when cloud copies are present.
- **All local data files must go inside `data_files/`** — corpus files → `data_files/corpus/`, chunked output → `data_files/chunked_output/`, CourtListener downloads/filtered → `data_files/cl_downloads/` and `data_files/cl_filtered/`. Never write data files to the project root or any directory outside `data_files/`.
- All ingest scripts use `--local-only` flag (consistent across all scripts); omit it to also upload to S3
- Use `python3` (not `python`) on this system
- The scraper rate-limits at 0.75s/request by default (`--delay` flag on `ilga_ingest.py`)
- `data_files/corpus/ilcs_corpus.jsonl.done_acts` is the checkpoint file for resuming ILCS scraping
- `data_files/corpus/cap_bulk_corpus.jsonl.done` is the checkpoint file for resuming CAP bulk download (case-ID granularity)
- BM25 index is disk-cached at `data_files/bm25_cache/` (~1.2 GB). Loads from cache on startup if row counts match `meta.json`; rebuilds from Supabase automatically when counts change. Loads in a background thread — the app is immediately usable (vector-only) while BM25 warms up (~15–45s from cache on the server). The cache persists across `systemctl restart`; only re-upload to the server if it is reprovisioned.
- **Retrieval state is eagerly pre-warmed** (`mcp_server/server.py`): a daemon thread starts `_get_state()` at module import time (i.e. when Streamlit starts) so the CrossEncoder model, Supabase collection probes, and BM25 background load are all in flight before the first user search. The 5 collection probes run in parallel via `ThreadPoolExecutor`. Without this, the first search would block for several seconds on cold init.
- Ollama must be running locally with `nomic-embed-text` pulled for embedding, and `llama3.2` for inference
- All sources (ILCS, ISCR, CAP, CourtListener, IAC, IDOC, SPAC, ICCB, Federal, Restore Justice, Cook County PD) are embedded in Supabase as of 2026-05-11

## Agent skills

### Issue tracker

Issues live in GitHub (`raphaeldchen/EJP-RAG`). See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role label vocabulary. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context — one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
