# Multi-Collection Retrieval Design

**Date:** 2026-05-03
**Status:** Approved

## Problem

The retrieval system was built for two sources (ILCS + ISCR). The corpus now has five source groups — statutes, court rules, court opinions, regulations/directives, and policy/advocacy documents — embedded into three new Supabase tables (`opinion_chunks`, `regulation_chunks`, `document_chunks`). The retrieval system doesn't know about any of them.

## Scope

Six files change. Everything else (postprocessor, query_engine, eval, embed, chunk) is untouched.

| File | Change |
|------|--------|
| `retrieval/config.py` | Add `CollectionConfig` dataclass + `COLLECTIONS` registry |
| `retrieval/bm25_store.py` | Iterate registry; swap `rank_bm25` → `bm25s`; add disk cache |
| `retrieval/indexes.py` | Rename `DualFusionRetriever` → `MultiCollectionRetriever`; hold `list[FusionRetriever]` |
| `retrieval/reflection.py` | Expand system prompt corpus description to all 5 collection types |
| `retrieval/main.py` | Unify citation extraction via `display_citation`; update test queries |
| `retrieval/query_engine.py` | Type hint update only |

The `_secondary_query` mutation pattern is out of scope — tracked separately in CLAUDE.md.

## Design

### 1. CollectionConfig registry (`config.py`)

```python
@dataclass
class CollectionConfig:
    id: str      # "ilcs" | "iscr" | "opinions" | "regulations" | "documents"
    table: str   # Supabase table name
    rpc: str     # vector search RPC function name

COLLECTIONS: list[CollectionConfig] = [
    CollectionConfig("ilcs",        "ilcs_chunks",        "match_ilcs_chunks"),
    CollectionConfig("iscr",        "court_rule_chunks",   "match_court_rule_chunks"),
    CollectionConfig("opinions",    "opinion_chunks",      "match_opinion_chunks"),
    CollectionConfig("regulations", "regulation_chunks",   "match_regulation_chunks"),
    CollectionConfig("documents",   "document_chunks",     "match_document_chunks"),
]
```

The `ilcs` and `iscr` entries read `table` and `rpc` from the existing `ILCS_TABLE`, `ILCS_RPC`, `ISCR_TABLE`, `ISCR_RPC` env vars (for experiment table overrides). The other three entries are not env-var-overridable for now.

The existing `ILCS_TABLE`, `ILCS_RPC`, `ISCR_TABLE`, `ISCR_RPC` module-level constants are kept for backward compatibility with `indexes.py`'s citation pinning path.

### 2. BM25: bm25s + disk cache (`bm25_store.py`)

**Why bm25s:** `opinion_chunks` adds ~337k chunks. Loading that from Supabase on every startup would take 3–5 minutes. `bm25s` uses numpy/scipy sparse internals and supports memory-mapped loading — the cache file is mapped into memory at startup (near-instant), and pages are read from disk only as queries access them.

**Startup flow:**

```
1. Fetch COUNT(*) from each of the 5 tables  (~5 fast queries, milliseconds)
2. If data_files/bm25_cache/meta.json exists and counts match → load from cache (mmap)
3. Otherwise → fetch all rows from Supabase, tokenize, build index, save cache + meta.json
```

**Cache layout:**
```
data_files/bm25_cache/        (gitignored)
  meta.json                   {"ilcs": 8420, "iscr": 1203, "opinions": 337505, ...}
  *.npz                       bm25s index files (numpy arrays)
  corpus_data.pkl             chunk_ids, texts, enriched_texts, metadata lists
```

**What gets indexed:** plain `text` for all 5 collections (same rationale as before — enriched_text header inflation skews BM25 scores). `enriched_text` stored alongside for the reranker and LLM.

**Score filtering:** the existing `if scores[idx] == 0: continue` filter is replaced by bm25s's natural top-k behavior (zero-score documents are not returned).

**Manual rebuild:** `python3 -m retrieval.bm25_build` — forces a full rebuild regardless of counts.

**Score calibration note:** bm25s uses BM25+ (a variant preventing zero scores for docs containing all query terms). Score values differ from rank_bm25's BM25Okapi. Run eval before and after to confirm ranking quality is equivalent or better.

### 3. MultiCollectionRetriever (`indexes.py`)

`DualFusionRetriever` is renamed `MultiCollectionRetriever`. Named attributes `_ilcs`, `_iscr` become `_retrievers: list[FusionRetriever]`.

```python
class MultiCollectionRetriever(BaseRetriever):
    def __init__(self, retrievers: list[FusionRetriever]):
        self._retrievers = retrievers
        self._secondary_query: str | None = None
        super().__init__()

    def _retrieve(self, query_bundle) -> list[NodeWithScore]:
        for r in self._retrievers:
            r._secondary_query = self._secondary_query
        try:
            results = [r._retrieve(query_bundle) for r in self._retrievers]
        finally:
            for r in self._retrievers:
                r._secondary_query = None
        return merge_ranked_lists(results, top_n=40)  # equal weights across all collections
```

`build_all_retrievers` iterates `COLLECTIONS`. `build_dual_retriever` is renamed `build_multi_retriever`.

**Collection weighting:** equal weights for now. Differential weighting (authoritative sources > advisory documents) is a future exploration item — tracked in CLAUDE.md.

**Citation pinning** (`_fetch_by_citation`) is unchanged — it still looks up only `ilcs_chunks` by `section_citation`. ILCS citations appear in queries far more than other source types and the lookup is intentionally source-specific.

### 4. Reflection prompt (`reflection.py`)

The corpus description block expands from 2 to 5 source types. Existing ILCS and ISCR descriptions are unchanged. New additions:

- **Court opinions:** IL Supreme Court + Appellate (1973–2024, CAP), 7th Circuit (CourtListener). Use for judicial interpretation, constitutional challenges, sentencing precedent.
- **Regulations and directives:** IL Admin Code Title 20 (519 IDOC-relevant sections), IDOC Administrative Directives (103 records) + reentry resources. Use for facility rules, programming requirements, disciplinary procedures, reentry planning.
- **Policy and advocacy documents:** SPAC publications, ICCB correctional education reports (FY2020–2025), federal docs (Federal Register, BOP policy, ED Dear Colleague Letters), Restore Justice IL, Cook County Public Defender resources. Use for sentencing policy trends, correctional education data, federal/state intersection, advocacy resources.

Classification and rewriting instructions are unchanged.

### 5. Citation extraction (`main.py`)

`_extract_citations()` is unified through `display_citation` with a legacy fallback for the two tables that predate the shared Chunk schema:

```
1. Try display_citation (all new tables: opinion_chunks, regulation_chunks, document_chunks)
2. Fall back to section_citation (ilcs_chunks — legacy column)
3. Fall back to rule_number + rule_title (court_rule_chunks — legacy column)
```

Three new test queries added:
- IL Supreme Court proportionality review for extended-term sentences (opinions)
- IDOC policy on disciplinary segregation and educational programming (regulations)
- SPAC data on racial composition of IL prison admissions for drug offenses (documents)

## Out of scope

- `_secondary_query` mutation pattern (CLAUDE.md code quality backlog)
- Adding `display_citation` column to `ilcs_chunks` / `court_rule_chunks` (separate migration)
- Differential collection weighting in RRF (future exploration, noted in CLAUDE.md)
- Graph DB integration (separate initiative)
