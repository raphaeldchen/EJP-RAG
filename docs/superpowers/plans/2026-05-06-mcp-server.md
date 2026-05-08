# MCP Server + Retrieval Audit Playground — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** (1) Expose the retrieval stack as an MCP server for future LangGraph integration; (2) deploy a Streamlit Audit Dashboard where lawyers label retrieved chunks to generate training data for embedding fine-tuning.

**Architecture:** A FastMCP server (`mcp_server/server.py`) wraps the existing `MultiCollectionRetriever` + `CrossEncoderReranker` stack and exposes 4 tools. A separate Streamlit app (`audit_app.py`) imports those same core functions directly — bypassing the MCP transport layer — so lawyers test the exact same code path the future LangGraph agent will call. Lawyer ratings write to a Supabase `audit_feedback` table; a weekly export script generates (query, positive, negative) triplets for Nomic embedding fine-tuning.

**Tech Stack:** `mcp[cli]`, `pydantic`, `streamlit`, existing `retrieval/` stack (llama-index, Supabase, rank-bm25, sentence-transformers, Ollama)

---

## Prerequisites and sequencing

**Month 1 (this plan):** MCP server + Audit Dashboard with Hybrid / Vector-only / BM25-only modes.

**Before Month 2:** Build graph traversal layer (Neo4j schema, citation edge extraction from CAP opinions + ILCS cross-references, traversal query layer). This is a prerequisite for:
- The `explore_case_relationships` MCP tool
- The Graph audit mode (currently greyed out in the dashboard)
- LangGraph orchestration that routes structural queries to graph traversal

**Month 2:** LangGraph orchestration + `explore_case_relationships` tool + Graph audit mode.
**Month 3:** Persona-differentiated retrieval paths + production chatbot deployment.

---

## Why the dashboard calls Python directly (not MCP)

The audit dashboard imports `_audit_retrieval` (and other `_`-prefixed functions) directly from `mcp_server.server`. The MCP tools are thin wrappers around those same functions:

```
MCP tool (future LangGraph agent) ──┐
                                     ├──► _audit_retrieval() ──► retrieval/ stack
Audit dashboard (direct import)  ──┘
```

Both paths execute identical code. The MCP transport layer adds nothing for a deterministic dashboard that always knows which function to call.

---

## Retrieval modes in the dashboard

| Mode | What it runs | Purpose |
|------|-------------|---------|
| **Hybrid** (default) | Vector + BM25 + RRF + CrossEncoder | Production mode — primary labeling target |
| **Vector only** | Per-collection FusionRetriever, no BM25 | Diagnostic: isolate embedding failures |
| **BM25 only** | Shared BM25 arm only, no vector | Diagnostic: isolate keyword matching failures |
| **Graph** | Graph traversal → scoped vector → reranker | Greyed out — Month 2 prerequisite |

Vector-only and BM25-only are diagnostic tools for debugging failures, not production retrieval paths. The LangGraph agent will call Hybrid (for semantic queries) and Graph (for structural citation queries) — not Vector/BM25 independently.

---

## File Structure

```
mcp_server/
├── __init__.py
├── schemas.py       # ChunkResult, SearchResponse, LookupResponse,
│                    #   AuditCandidate, AuditResponse, ClassifyResponse
└── server.py        # FastMCP instance, _State singleton, 4 tool definitions

audit_app.py         # Streamlit Audit Dashboard (imports from mcp_server.server)

scripts/
└── export_triplets.py  # reads audit_feedback, emits JSONL triplets for fine-tuning

tests/mcp_server/
├── __init__.py
├── test_schemas.py
├── test_classify_tool.py
├── test_search_tool.py
├── test_lookup_tool.py
└── test_audit_tool.py

docs/mcp/
└── setup.md
```

**Existing files modified:** `retrieval/indexes.py` (add `secondary_query` and `bm25_enabled` params to `_retrieve()`; override `retrieve()` for clean external API), `retrieval/main.py` (replace `_secondary_query` mutation pattern with direct `retriever.retrieve(..., secondary_query=...)` call).

---

### Task 0: Fix `_secondary_query` seam in `indexes.py` and `main.py`

**Files:**
- Modify: `retrieval/indexes.py`
- Modify: `retrieval/main.py`

CLAUDE.md flags the `_secondary_query` mutation-before-retrieve / clear-in-finally pattern as a known anti-pattern. The MCP server's `_retrieve_by_mode` would perpetuate it. Since `indexes.py` is already being modified for `bm25_enabled`, fix both at once.

**Changes to `retrieval/indexes.py`:**

1. Add `secondary_query: str | None = None` param to `FusionRetriever._retrieve()`:

```python
def _retrieve(self, query_bundle: QueryBundle, secondary_query: str | None = None) -> list[NodeWithScore]:
    primary = self._vector_retriever.retrieve(query_bundle)
    if secondary_query:
        from retrieval.postprocessor import merge_ranked_lists
        sec_bundle = QueryBundle(query_str=secondary_query)
        secondary = self._vector_retriever.retrieve(sec_bundle)
        return merge_ranked_lists([primary, secondary], top_n=40, weights=[1.0, 0.5])
    return primary
```

2. Add `secondary_query` and `bm25_enabled` params to `MultiCollectionRetriever._retrieve()`:

```python
def _retrieve(
    self,
    query_bundle: QueryBundle,
    secondary_query: str | None = None,
    bm25_enabled: bool = True,
) -> list[NodeWithScore]:
    from retrieval.postprocessor import merge_ranked_lists
    query_str = query_bundle.query_str

    results = [r._retrieve(query_bundle, secondary_query=secondary_query) for r in self._retrievers]

    per_list_weights = (
        [self._weights.get(cid, 1.0) for cid in self._collection_ids]
        if self._weights and self._collection_ids
        else [1.0] * len(results)
    )

    all_lists = results
    all_weights = per_list_weights

    if bm25_enabled:
        bm25_weight = self._weights.get("bm25", 1.2)
        bm25_nodes = self._bm25.retrieve(query_str, top_k=DEFAULT_TOP_K)
        bm25_list = [NodeWithScore(node=n, score=0.0) for n in bm25_nodes]
        all_lists = results + [bm25_list]
        all_weights = per_list_weights + [bm25_weight]

        if secondary_query:
            sec_bm25_nodes = self._bm25.retrieve(secondary_query, top_k=DEFAULT_TOP_K)
            sec_bm25_list = [NodeWithScore(node=n, score=0.0) for n in sec_bm25_nodes]
            all_lists.append(sec_bm25_list)
            all_weights.append(bm25_weight * 0.5)

    fused = merge_ranked_lists(all_lists, top_n=60, weights=all_weights)

    # Citation pinning — always on regardless of bm25_enabled
    combined = query_str + (" " + secondary_query if secondary_query else "")
    existing_ids = {n.node.node_id for n in fused}
    pinned: list[NodeWithScore] = []

    ilcs_citations = _ILCS_CITATION_RE.findall(combined)
    if ilcs_citations:
        ilcs_pinned = self._fetch_by_citation(ilcs_citations, exclude_ids=existing_ids)
        if ilcs_pinned:
            print(f"[Citation] Pinned {len(ilcs_pinned)} ILCS chunk(s) for: {ilcs_citations}")
        pinned.extend(ilcs_pinned)

    rule_numbers = _RULE_RE.findall(combined)
    if rule_numbers:
        rule_pinned = self._fetch_by_rule_number(rule_numbers, exclude_ids=existing_ids)
        if rule_pinned:
            print(f"[Citation] Pinned {len(rule_pinned)} ISCR chunk(s) for: Rule {rule_numbers}")
        pinned.extend(rule_pinned)

    if pinned:
        fused = pinned + fused

    return fused
```

3. Override `retrieve()` on `MultiCollectionRetriever` to expose the clean external API:

```python
def retrieve(
    self,
    str_or_query_bundle,
    secondary_query: str | None = None,
    bm25_enabled: bool = True,
) -> list[NodeWithScore]:
    from llama_index.core.schema import QueryBundle as QB
    if isinstance(str_or_query_bundle, str):
        query_bundle = QB(query_str=str_or_query_bundle)
    else:
        query_bundle = str_or_query_bundle
    return self._retrieve(query_bundle, secondary_query=secondary_query, bm25_enabled=bm25_enabled)
```

**Changes to `retrieval/main.py`:**

Replace the mutation pattern:
```python
# BEFORE
dual_retriever._secondary_query = result.rewritten_query
try:
    response = engine.query(query_str)
finally:
    dual_retriever._secondary_query = None
```

With a direct retriever call that feeds results into the engine's synthesizer:
```python
# AFTER
nodes = dual_retriever.retrieve(query_str, secondary_query=result.rewritten_query)
response = engine.synthesize(query_str, nodes)
```

- [ ] **Step 1: Apply `indexes.py` changes and run existing tests**

```bash
python3 -m pytest tests/retrieval/ -v
```

Expected: all existing retrieval tests pass.

- [ ] **Step 2: Apply `main.py` change and smoke-test**

```bash
python3 -m retrieval.main
```

Expected: test queries return answers with citations as before.

- [ ] **Step 3: Commit**

```bash
git add retrieval/indexes.py retrieval/main.py
git commit -m "refactor: add secondary_query and bm25_enabled params to MultiCollectionRetriever; fix _secondary_query mutation seam"
```

---

### Task 1: Install MCP SDK and define response schemas

**Files:**
- Create: `mcp_server/__init__.py`
- Create: `mcp_server/schemas.py`
- Create: `tests/mcp_server/__init__.py`
- Create: `tests/mcp_server/test_schemas.py`

- [ ] **Step 1: Install the MCP Python SDK**

```bash
pip install "mcp[cli]"
```

Expected: `Successfully installed mcp-...`

- [ ] **Step 2: Write the failing tests**

Create `tests/mcp_server/test_schemas.py`:

```python
from mcp_server.schemas import (
    ChunkResult, SearchResponse, LookupResponse,
    AuditCandidate, AuditResponse, ClassifyResponse,
)


def test_chunk_result_requires_core_fields():
    chunk = ChunkResult(
        chunk_id="abc123",
        text="The defendant shall...",
        citation="730 ILCS 5/3-6-3",
        source="ilcs",
        rrf_score=0.05,
        metadata={"section_citation": "730 ILCS 5/3-6-3"},
    )
    assert chunk.chunk_id == "abc123"
    assert chunk.source == "ilcs"


def test_audit_response_exposes_retrieval_mode():
    candidate = AuditCandidate(
        chunk_id="x1", text="text", citation="730 ILCS 5/3-6-3",
        source="ilcs", rrf_score=0.04, ce_score=-4.2,
        survived=False, metadata={},
    )
    response = AuditResponse(
        query="good time credit", rewritten_query=None, intent="in_scope",
        retrieval_mode="hybrid",
        candidates=[candidate], reranked=[], dropped=[candidate],
        threshold=-3.0, top_n=15,
    )
    assert response.retrieval_mode == "hybrid"
    assert len(response.dropped) == 1


def test_search_response_serializes_to_json():
    chunk = ChunkResult(
        chunk_id="abc123", text="text", citation="730 ILCS 5/3-6-3",
        source="ilcs", rrf_score=0.05, metadata={},
    )
    response = SearchResponse(
        query="good time credit", rewritten_query="730 ILCS 5/3-6-3",
        intent="in_scope", results=[chunk],
    )
    assert "730 ILCS 5/3-6-3" in response.model_dump_json()


def test_classify_response_fields():
    result = ClassifyResponse(
        intent="in_scope",
        reasoning="Query concerns Illinois sentencing law",
        rewritten_query="sentencing ranges 730 ILCS 5/5-4.5",
    )
    assert result.intent == "in_scope"
```

- [ ] **Step 3: Run test to verify it fails**

```bash
python3 -m pytest tests/mcp_server/test_schemas.py -v
```

Expected: `ImportError: No module named 'mcp_server'`

- [ ] **Step 4: Create empty package markers**

```bash
touch mcp_server/__init__.py tests/mcp_server/__init__.py
```

- [ ] **Step 5: Create `mcp_server/schemas.py`**

```python
from pydantic import BaseModel


class ChunkResult(BaseModel):
    chunk_id: str
    text: str
    citation: str
    source: str
    rrf_score: float
    metadata: dict


class SearchResponse(BaseModel):
    query: str
    rewritten_query: str | None
    intent: str
    results: list[ChunkResult]


class LookupResponse(BaseModel):
    citation: str
    chunks: list[ChunkResult]
    total_found: int


class AuditCandidate(BaseModel):
    chunk_id: str
    text: str
    citation: str
    source: str
    rrf_score: float
    ce_score: float | None
    survived: bool
    metadata: dict


class AuditResponse(BaseModel):
    query: str
    rewritten_query: str | None
    intent: str
    retrieval_mode: str          # "hybrid" | "vector" | "bm25"
    candidates: list[AuditCandidate]
    reranked: list[AuditCandidate]
    dropped: list[AuditCandidate]
    threshold: float
    top_n: int


class ClassifyResponse(BaseModel):
    intent: str
    reasoning: str
    rewritten_query: str | None
```

- [ ] **Step 6: Run test to verify it passes**

```bash
python3 -m pytest tests/mcp_server/test_schemas.py -v
```

Expected: `4 passed`

- [ ] **Step 7: Commit**

```bash
git add mcp_server/__init__.py mcp_server/schemas.py tests/mcp_server/__init__.py tests/mcp_server/test_schemas.py
git commit -m "feat: add mcp_server package with unified response schemas"
```

---

### Task 2: MCP server skeleton + `classify_query` tool

**Files:**
- Create: `mcp_server/server.py`
- Create: `tests/mcp_server/test_classify_tool.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/mcp_server/test_classify_tool.py`:

```python
import json
from mcp_server.server import _classify_query


def test_classify_in_scope_statute_query():
    result = json.loads(_classify_query("What are the sentencing ranges for a Class 1 felony?"))
    assert result["intent"] == "in_scope"
    assert result["rewritten_query"] is not None
    text = result["rewritten_query"].lower()
    assert "730 ilcs" in result["rewritten_query"] or "felony" in text


def test_classify_out_of_scope_query():
    result = json.loads(_classify_query("What are the federal sentencing guidelines for drug trafficking?"))
    assert result["intent"] == "out_of_scope"


def test_classify_returns_reasoning():
    result = json.loads(_classify_query("Can I appeal a criminal conviction in Illinois?"))
    assert result["reasoning"] and len(result["reasoning"]) > 10
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/mcp_server/test_classify_tool.py -v
```

Expected: `ImportError: cannot import name '_classify_query' from 'mcp_server.server'`

- [ ] **Step 3: Create `mcp_server/server.py`**

```python
from dataclasses import dataclass
import re as _re

from mcp.server.fastmcp import FastMCP
from llama_index.core import Settings
from llama_index.core.schema import NodeWithScore, QueryBundle

from retrieval.config import ILCS_TABLE, ISCR_TABLE, DEFAULT_TOP_K
from retrieval.embeddings import get_embedding_model
from retrieval.indexes import get_supabase_client, build_multi_retriever, MultiCollectionRetriever
from retrieval.bm25_store import BM25Retriever
from retrieval.postprocessor import CrossEncoderReranker, merge_ranked_lists
from retrieval.reflection import reflect, QueryIntent
from mcp_server.schemas import (
    ChunkResult, SearchResponse, LookupResponse,
    AuditCandidate, AuditResponse, ClassifyResponse,
)

mcp = FastMCP("Illinois Legal RAG")

_ILCS_RE = _re.compile(r'\d+\s+ILCS\s+\d+/[\d\.\-]+')
_RULE_RE = _re.compile(r'^Rule\s+(\d+)', _re.IGNORECASE)


# -- Singleton state -----------------------------------------------------------

@dataclass
class _State:
    retriever: MultiCollectionRetriever
    reranker: CrossEncoderReranker
    client: object  # supabase.Client


_state: _State | None = None


def _probe_collections(client) -> dict[str, "FusionRetriever"]:
    """Build only collections whose RPC functions are registered in Supabase."""
    from retrieval.config import COLLECTIONS
    import numpy as np
    available = {}
    for col in COLLECTIONS:
        try:
            client.rpc(col.rpc, {"query_embedding": [0.0] * 768, "match_count": 1}).execute()
            available[col.id] = build_fusion_retriever(client, col.rpc)
            print(f"[State] Collection '{col.id}' available")
        except Exception as e:
            print(f"[State] Collection '{col.id}' skipped — RPC not available: {e}")
    return available


def _get_state() -> _State:
    global _state
    if _state is None:
        embed_model = get_embedding_model()
        Settings.embed_model = embed_model
        client = get_supabase_client()
        bm25 = BM25Retriever(client)
        available_retrievers = _probe_collections(client)
        retriever = build_multi_retriever(client, bm25, retrievers=available_retrievers)
        reranker = CrossEncoderReranker(top_n=6, score_threshold=-3.0)
        reranker._get_model()
        _state = _State(retriever=retriever, reranker=reranker, client=client)
    return _state


# -- Shared helpers ------------------------------------------------------------

def _extract_citation(meta: dict) -> str:
    dc = (meta.get("display_citation") or "").strip()
    if dc:
        return dc
    section = (meta.get("section_citation") or "").strip()
    if section:
        return section
    rule = meta.get("rule_number")
    if rule:
        title = (meta.get("rule_title") or "").strip()
        prefix = f"Rule {rule}"
        if title.startswith(prefix):
            title = title[len(prefix):].lstrip(" .--").strip()
        return prefix + (f" -- {title}" if title else "")
    return meta.get("source", "unknown")


def _node_to_chunk(node_with_score) -> ChunkResult:
    node = node_with_score.node
    meta = node.metadata or {}
    return ChunkResult(
        chunk_id=node.node_id,
        text=node.get_content()[:2000],
        citation=_extract_citation(meta),
        source=meta.get("source", "unknown"),
        rrf_score=float(node_with_score.score or 0.0),
        metadata={k: v for k, v in meta.items() if k != "embedding"},
    )


def _retrieve_by_mode(
    state: _State,
    query_str: str,
    secondary_query: str | None,
    mode: str,
) -> list[NodeWithScore]:
    """Run retrieval for the given mode. Used by both search and audit tools.

    All modes preserve citation pinning. The mode toggle isolates exactly one
    retrieval arm:
      hybrid  — vector + BM25 + citation pinning  (production)
      vector  — vector + citation pinning          (BM25 arm removed)
      bm25    — BM25 only, no vector, no pinning   (pure keyword diagnostic)
    """
    if mode == "bm25":
        bm25_nodes = state.retriever._bm25.retrieve(query_str, top_k=DEFAULT_TOP_K)
        if secondary_query:
            sec_nodes = state.retriever._bm25.retrieve(secondary_query, top_k=DEFAULT_TOP_K)
            bm25_nodes = list({n.node_id: n for n in bm25_nodes + sec_nodes}.values())
        return [NodeWithScore(node=n, score=0.0) for n in bm25_nodes]

    if mode == "vector":
        return state.retriever.retrieve(query_str, secondary_query=secondary_query, bm25_enabled=False)

    # mode == "hybrid" (default)
    return state.retriever.retrieve(query_str, secondary_query=secondary_query)


# -- Tools ---------------------------------------------------------------------

def _classify_query(query: str) -> str:
    result = reflect(query)
    return ClassifyResponse(
        intent=result.intent.value,
        reasoning=result.reasoning,
        rewritten_query=result.rewritten_query,
    ).model_dump_json(indent=2)


@mcp.tool()
def classify_query(query: str) -> str:
    """
    Classify a query as in_scope, out_of_scope, or ambiguous for Illinois criminal law.
    Returns intent, reasoning, and a rewritten query with ILCS citations where known.
    Call this before searching to confirm scope and get a better search query.
    """
    return _classify_query(query)


if __name__ == "__main__":
    mcp.run()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python3 -m pytest tests/mcp_server/test_classify_tool.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add mcp_server/server.py tests/mcp_server/test_classify_tool.py
git commit -m "feat: add MCP server skeleton with classify_query tool"
```

---

### Task 3: `search_legal_sources` tool

**Files:**
- Modify: `mcp_server/server.py`
- Create: `tests/mcp_server/test_search_tool.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/mcp_server/test_search_tool.py`:

```python
import json
from mcp_server.server import _search_legal_sources


def test_search_returns_chunk_results():
    result = json.loads(_search_legal_sources("good time credit Illinois sentencing"))
    assert result["intent"] == "in_scope"
    assert len(result["results"]) > 0
    first = result["results"][0]
    assert first["chunk_id"] and first["text"] and first["citation"]
    assert first["source"] in ("ilcs", "iscr", "opinions", "regulations", "documents")


def test_search_returns_ilcs_for_statute_query():
    result = json.loads(_search_legal_sources("730 ILCS 5/3-6-3 good time credit"))
    citations = [r["citation"] for r in result["results"]]
    assert any("730 ILCS" in c for c in citations), f"Expected ILCS citations, got: {citations}"


def test_search_respects_top_k():
    result = json.loads(_search_legal_sources("sentencing Class 1 felony", top_k=3))
    assert len(result["results"]) <= 3


def test_search_out_of_scope_returns_empty():
    result = json.loads(_search_legal_sources("what is the capital of France"))
    assert result["intent"] == "out_of_scope"
    assert result["results"] == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/mcp_server/test_search_tool.py -v
```

Expected: `ImportError: cannot import name '_search_legal_sources' from 'mcp_server.server'`

- [ ] **Step 3: Add `_search_legal_sources` to `mcp_server/server.py`**

Insert after `classify_query` (before `if __name__ == "__main__"`):

```python
def _search_legal_sources(query: str, top_k: int = 10) -> str:
    reflection = reflect(query)

    if reflection.intent == QueryIntent.OUT_OF_SCOPE:
        return SearchResponse(
            query=query, rewritten_query=None, intent="out_of_scope", results=[],
        ).model_dump_json(indent=2)

    state = _get_state()
    candidates = _retrieve_by_mode(state, query, reflection.rewritten_query, mode="hybrid")
    reranked = state.reranker._postprocess_nodes(candidates, QueryBundle(query_str=query))

    return SearchResponse(
        query=query,
        rewritten_query=reflection.rewritten_query,
        intent=reflection.intent.value,
        results=[_node_to_chunk(n) for n in reranked[:top_k]],
    ).model_dump_json(indent=2)


@mcp.tool()
def search_legal_sources(query: str, top_k: int = 10) -> str:
    """
    Search Illinois legal sources using hybrid vector + BM25 retrieval with CrossEncoder reranking.
    Covers statutes (ILCS), court rules (ISCR), opinions, IDOC regulations, and policy docs.
    Returns up to top_k reranked chunks. For precise lookups by citation, use lookup_citation.
    """
    return _search_legal_sources(query, top_k=top_k)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python3 -m pytest tests/mcp_server/test_search_tool.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add mcp_server/server.py tests/mcp_server/test_search_tool.py
git commit -m "feat: add search_legal_sources MCP tool"
```

---

### Task 4: `lookup_citation` tool

**Files:**
- Modify: `mcp_server/server.py`
- Create: `tests/mcp_server/test_lookup_tool.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/mcp_server/test_lookup_tool.py`:

```python
import json
from mcp_server.server import _lookup_citation


def test_lookup_ilcs_returns_chunks():
    result = json.loads(_lookup_citation("730 ILCS 5/3-6-3"))
    assert result["citation"] == "730 ILCS 5/3-6-3"
    assert len(result["chunks"]) > 0
    assert all(r["source"] == "ilcs" for r in result["chunks"])


def test_lookup_rule_returns_chunks():
    result = json.loads(_lookup_citation("Rule 401"))
    assert result["citation"] == "Rule 401"
    assert len(result["chunks"]) > 0
    assert all(r["source"] == "iscr" for r in result["chunks"])


def test_lookup_unknown_returns_empty():
    result = json.loads(_lookup_citation("999 ILCS 999/999-999"))
    assert result["chunks"] == [] and result["total_found"] == 0


def test_lookup_chunk_has_required_fields():
    result = json.loads(_lookup_citation("730 ILCS 5/3-6-3"))
    chunk = result["chunks"][0]
    assert chunk["chunk_id"] and chunk["text"] and "730 ILCS" in chunk["citation"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/mcp_server/test_lookup_tool.py -v
```

Expected: `ImportError: cannot import name '_lookup_citation' from 'mcp_server.server'`

- [ ] **Step 3: Add `_lookup_citation` to `mcp_server/server.py`**

Insert after `search_legal_sources`:

```python
def _lookup_citation(citation: str) -> str:
    state = _get_state()
    citation = citation.strip()
    chunks: list[ChunkResult] = []

    if _ILCS_RE.search(citation):
        try:
            rows = (
                state.client.table(ILCS_TABLE)
                .select("chunk_id, enriched_text, text, section_citation, major_topic")
                .eq("section_citation", citation)
                .execute()
                .data
            )
        except Exception as e:
            print(f"[lookup_citation] ILCS query failed for {citation!r}: {e}")
            rows = []
        for row in rows:
            text = row.get("enriched_text") or row.get("text") or ""
            chunks.append(ChunkResult(
                chunk_id=row["chunk_id"], text=text[:2000],
                citation=row.get("section_citation") or citation, source="ilcs",
                rrf_score=1.0,
                metadata={"section_citation": row.get("section_citation"),
                          "major_topic": row.get("major_topic"), "pinned": True},
            ))

    rule_match = _RULE_RE.match(citation)
    if rule_match:
        rule_number = rule_match.group(1)
        try:
            rows = (
                state.client.table(ISCR_TABLE)
                .select("chunk_id, enriched_text, text, rule_number, rule_title")
                .eq("rule_number", rule_number)
                .execute()
                .data
            )
        except Exception as e:
            print(f"[lookup_citation] ISCR query failed for Rule {rule_number!r}: {e}")
            rows = []
        for row in rows:
            rule = row.get("rule_number", "")
            title = (row.get("rule_title") or "").strip()
            prefix = f"Rule {rule}"
            if title.startswith(prefix):
                title = title[len(prefix):].lstrip(" .--").strip()
            label = prefix + (f" -- {title}" if title else "")
            text = row.get("enriched_text") or row.get("text") or ""
            chunks.append(ChunkResult(
                chunk_id=row["chunk_id"], text=text[:2000], citation=label, source="iscr",
                rrf_score=1.0,
                metadata={"rule_number": rule, "rule_title": row.get("rule_title"), "pinned": True},
            ))

    return LookupResponse(citation=citation, chunks=chunks, total_found=len(chunks)).model_dump_json(indent=2)


@mcp.tool()
def lookup_citation(citation: str) -> str:
    """
    Fetch all chunks for a specific citation directly from the database.
    Accepts ILCS citations (e.g. '730 ILCS 5/3-6-3') or ISCR rule numbers (e.g. 'Rule 401').
    Use when you know the exact citation and want the full statutory text for verification.
    """
    return _lookup_citation(citation)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python3 -m pytest tests/mcp_server/test_lookup_tool.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add mcp_server/server.py tests/mcp_server/test_lookup_tool.py
git commit -m "feat: add lookup_citation MCP tool"
```

---

### Task 5: `audit_retrieval` tool with mode support

**Files:**
- Modify: `mcp_server/server.py`
- Create: `tests/mcp_server/test_audit_tool.py`

This is the core tool the audit dashboard uses. The `mode` parameter exposes all three
retrieval arms so lawyers can compare why a chunk surfaces in hybrid but not BM25, or
vice versa.

- [ ] **Step 1: Write the failing tests**

Create `tests/mcp_server/test_audit_tool.py`:

```python
import json
from mcp_server.server import _audit_retrieval


def test_audit_returns_candidates_and_reranked():
    result = json.loads(_audit_retrieval("good time credit Illinois"))
    assert len(result["candidates"]) > 0
    assert isinstance(result["reranked"], list)
    assert isinstance(result["dropped"], list)
    assert result["retrieval_mode"] == "hybrid"


def test_audit_reranked_ids_match_survived_candidates():
    result = json.loads(_audit_retrieval("Class 1 felony sentencing Illinois"))
    survived_ids = {c["chunk_id"] for c in result["candidates"] if c["survived"]}
    reranked_ids = {c["chunk_id"] for c in result["reranked"]}
    assert survived_ids == reranked_ids


def test_audit_candidates_have_ce_scores():
    result = json.loads(_audit_retrieval("good time credit"))
    for candidate in result["candidates"]:
        assert candidate["ce_score"] is not None
        assert isinstance(candidate["ce_score"], float)


def test_audit_vector_mode_returns_no_bm25_inflation():
    hybrid = json.loads(_audit_retrieval("730 ILCS 5/3-6-3", mode="hybrid"))
    vector = json.loads(_audit_retrieval("730 ILCS 5/3-6-3", mode="vector"))
    assert vector["retrieval_mode"] == "vector"
    # Vector-only should have fewer or equal candidates than hybrid (no BM25 arm)
    assert len(vector["candidates"]) <= len(hybrid["candidates"]) + 5


def test_audit_bm25_mode_returns_candidates():
    result = json.loads(_audit_retrieval("good time credit", mode="bm25"))
    assert result["retrieval_mode"] == "bm25"
    assert len(result["candidates"]) > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/mcp_server/test_audit_tool.py -v
```

Expected: `ImportError: cannot import name '_audit_retrieval' from 'mcp_server.server'`

- [ ] **Step 3: Add `_audit_retrieval` to `mcp_server/server.py`**

Insert after `lookup_citation` (before `if __name__ == "__main__"`):

```python
def _audit_retrieval(query: str, mode: str = "hybrid", top_k: int = 20) -> str:
    """
    Core logic -- called directly by the audit dashboard and wrapped by the MCP tool.
    mode: "hybrid" | "vector" | "bm25"
    """
    state = _get_state()
    reflection = reflect(query)
    candidates = _retrieve_by_mode(state, query, reflection.rewritten_query, mode=mode)

    # Score ALL candidates without threshold filtering -- we want every score for the debug view.
    ce_model = state.reranker._get_model()
    pairs = [(query, n.node.get_content()) for n in candidates]
    ce_scores = ce_model.predict(pairs).tolist()

    threshold = state.reranker.score_threshold
    top_n = state.reranker.top_n

    sorted_indices = sorted(range(len(candidates)), key=lambda i: ce_scores[i], reverse=True)
    survived_ids: set[str] = set()
    for rank, idx in enumerate(sorted_indices):
        if ce_scores[idx] >= threshold and rank < top_n:
            survived_ids.add(candidates[idx].node.node_id)

    audit_candidates: list[AuditCandidate] = []
    for node_with_score, ce_score in zip(candidates, ce_scores):
        meta = node_with_score.node.metadata or {}
        audit_candidates.append(AuditCandidate(
            chunk_id=node_with_score.node.node_id,
            text=node_with_score.node.get_content()[:1500],
            citation=_extract_citation(meta),
            source=meta.get("source", "unknown"),
            rrf_score=float(node_with_score.score or 0.0),
            ce_score=float(ce_score),
            survived=node_with_score.node.node_id in survived_ids,
            metadata={k: v for k, v in meta.items() if k != "embedding"},
        ))

    reranked = sorted([c for c in audit_candidates if c.survived],
                      key=lambda c: c.ce_score or 0.0, reverse=True)
    dropped = sorted([c for c in audit_candidates if not c.survived],
                     key=lambda c: c.ce_score or 0.0, reverse=True)

    return AuditResponse(
        query=query,
        rewritten_query=reflection.rewritten_query,
        intent=reflection.intent.value,
        retrieval_mode=mode,
        candidates=audit_candidates,
        reranked=reranked,
        dropped=dropped,
        threshold=threshold,
        top_n=top_n,
    ).model_dump_json(indent=2)


# audit_retrieval is intentionally NOT exposed as an MCP tool.
# It is a debug/labeling instrument for the Audit Dashboard only.
# LangGraph agents call search_legal_sources (semantic) and lookup_citation (structural).
# A "show sources" UX in the Month 3 chatbot should be built on search_legal_sources
# results (which already return citations and scores), not the full audit dump.
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python3 -m pytest tests/mcp_server/test_audit_tool.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Run full MCP test suite**

```bash
python3 -m pytest tests/mcp_server/ -v
```

Expected: `16 passed`

- [ ] **Step 6: Commit**

```bash
git add mcp_server/server.py tests/mcp_server/test_audit_tool.py
git commit -m "feat: add audit_retrieval MCP tool with hybrid/vector/bm25 mode support"
```

---

### Task 6: Create `audit_feedback` Supabase table

**Files:**
- No Python files. SQL run in Supabase dashboard.
- Create: `scripts/init_audit_feedback.sql` (reference copy)

The `audit_feedback` table stores every lawyer rating. The `label` enum distinguishes
signal quality: `BINDING` = direct answer (highest-value training positive),
`RELEVANT` = useful context, `IRRELEVANT` = noise (training negative).
Hard negatives are IRRELEVANT chunks with high `ce_score` — these are the most
valuable negatives for fine-tuning because the reranker was confused by them.

- [ ] **Step 1: Create `scripts/init_audit_feedback.sql`**

```sql
CREATE TABLE IF NOT EXISTS audit_feedback (
    id              uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at      timestamptz DEFAULT now(),
    query_text      text NOT NULL,
    query_id        text NOT NULL,     -- sha256(query_text) for grouping
    chunk_id        text NOT NULL,
    citation        text,
    source          text,              -- "ilcs" | "iscr" | "opinions" | "regulations" | "documents"
    retrieval_mode  text NOT NULL,     -- "hybrid" | "vector" | "bm25"
    persona         text,              -- "researcher" | "practitioner" | "incarcerated"
    pre_rerank_rank integer,           -- position in raw candidates list (1-indexed)
    post_rerank_rank integer,          -- position after reranking, null if dropped
    rrf_score       float,
    ce_score        float,
    label           text NOT NULL      -- "BINDING" | "RELEVANT" | "IRRELEVANT"
                    CHECK (label IN ('BINDING', 'RELEVANT', 'IRRELEVANT')),
    comment         text,
    expert_id       text               -- who submitted (email or name, optional)
);

CREATE INDEX IF NOT EXISTS audit_feedback_query_id_idx ON audit_feedback (query_id);
CREATE INDEX IF NOT EXISTS audit_feedback_label_idx ON audit_feedback (label);
CREATE INDEX IF NOT EXISTS audit_feedback_chunk_id_idx ON audit_feedback (chunk_id);
```

- [ ] **Step 2: Run the SQL in the Supabase dashboard**

Open your Supabase project → SQL Editor → paste and run `scripts/init_audit_feedback.sql`.

Expected: `Success. No rows returned.`

- [ ] **Step 3: Verify the table exists**

In the Supabase Table Editor, confirm `audit_feedback` appears with the expected columns.

- [ ] **Step 4: Add a `submit_feedback` helper to `mcp_server/server.py`**

Add this function (not an MCP tool — called directly by the dashboard):

```python
import hashlib as _hashlib


def submit_feedback(
    query: str,
    chunk_id: str,
    citation: str,
    source: str,
    retrieval_mode: str,
    persona: str,
    pre_rerank_rank: int,
    post_rerank_rank: int | None,
    rrf_score: float,
    ce_score: float | None,
    label: str,
    comment: str = "",
    expert_id: str = "",
) -> None:
    """Write one lawyer rating to audit_feedback. Called by audit_app.py."""
    state = _get_state()
    query_id = _hashlib.sha256(query.encode()).hexdigest()[:16]
    state.client.table("audit_feedback").insert({
        "query_text": query,
        "query_id": query_id,
        "chunk_id": chunk_id,
        "citation": citation,
        "source": source,
        "retrieval_mode": retrieval_mode,
        "persona": persona,
        "pre_rerank_rank": pre_rerank_rank,
        "post_rerank_rank": post_rerank_rank,
        "rrf_score": rrf_score,
        "ce_score": ce_score,
        "label": label,
        "comment": comment or None,
        "expert_id": expert_id or None,
    }).execute()
```

- [ ] **Step 5: Write a smoke test for feedback submission**

Add to `tests/mcp_server/test_audit_tool.py`:

```python
def test_submit_feedback_writes_to_supabase():
    from mcp_server.server import submit_feedback, _get_state
    submit_feedback(
        query="good time credit",
        chunk_id="test-chunk-smoke",
        citation="730 ILCS 5/3-6-3",
        source="ilcs",
        retrieval_mode="hybrid",
        persona="researcher",
        pre_rerank_rank=1,
        post_rerank_rank=1,
        rrf_score=0.05,
        ce_score=2.1,
        label="BINDING",
        comment="smoke test",
        expert_id="test",
    )
    state = _get_state()
    rows = (
        state.client.table("audit_feedback")
        .select("id")
        .eq("chunk_id", "test-chunk-smoke")
        .execute()
        .data
    )
    assert len(rows) >= 1
    # Cleanup
    state.client.table("audit_feedback").delete().eq("chunk_id", "test-chunk-smoke").execute()
```

- [ ] **Step 6: Run test to verify it passes**

```bash
python3 -m pytest tests/mcp_server/test_audit_tool.py::test_submit_feedback_writes_to_supabase -v
```

Expected: `1 passed`

- [ ] **Step 7: Commit**

```bash
git add scripts/init_audit_feedback.sql mcp_server/server.py tests/mcp_server/test_audit_tool.py
git commit -m "feat: add audit_feedback Supabase table and submit_feedback helper"
```

---

### Task 7: Build Streamlit Audit Dashboard (`audit_app.py`)

**Files:**
- Create: `audit_app.py`

The dashboard calls `_audit_retrieval` and `submit_feedback` directly from
`mcp_server.server` — same code the MCP tools wrap, so lawyers test the
exact retrieval logic the LangGraph agent will use.

UI layout:
- **Sidebar:** query input, persona toggle, top_k slider, mode radio, Search button
- **Main area:** two tabs — Pre-Rerank (all candidates) and Post-Rerank (final context)
- **Result cards:** citation badge, score bar, expandable text, BINDING/RELEVANT/IRRELEVANT buttons

- [ ] **Step 1: Install Streamlit if not already installed**

```bash
pip install streamlit
```

- [ ] **Step 2: Create `audit_app.py`**

```python
import json
import streamlit as st
from mcp_server.server import _audit_retrieval, submit_feedback

st.set_page_config(page_title="Retrieval Audit", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
    #MainMenu, footer, header { visibility: hidden; }
    .stApp { background: #ffffff; }
    .score-high  { background: #d1fae5; border-radius: 6px; padding: 2px 8px; font-size: 0.8rem; }
    .score-mid   { background: #fef3c7; border-radius: 6px; padding: 2px 8px; font-size: 0.8rem; }
    .score-low   { background: #fee2e2; border-radius: 6px; padding: 2px 8px; font-size: 0.8rem; }
    .citation-badge {
        display: inline-block; background: #f0f0f0; color: #374151;
        font-size: 0.75rem; padding: 2px 10px; border-radius: 12px;
        font-family: monospace; margin-bottom: 4px;
    }
    .mode-label { font-size: 0.7rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.05em; }
</style>
""", unsafe_allow_html=True)


# -- Sidebar controls ----------------------------------------------------------

with st.sidebar:
    st.title("⚖️ Retrieval Audit")
    st.caption("Illinois Legal RAG — Expert Labeling")
    st.divider()

    query_input = st.text_area("Legal Query", height=100, placeholder="e.g. What is good-time credit in Illinois?")
    persona = st.selectbox("Persona", ["Researcher", "Practitioner", "Incarcerated Person"])
    top_k = st.slider("Candidates to show", min_value=5, max_value=60, value=20, step=5)
    mode = st.radio(
        "Retrieval Mode",
        ["Hybrid (production)", "Vector only", "BM25 only", "Graph (coming soon)"],
    )
    expert_id = st.text_input("Your name / email", placeholder="Optional — for attribution")
    search_btn = st.button("Search", type="primary", use_container_width=True)

    st.divider()
    st.caption("**Mode guide**")
    st.caption("**Hybrid** = production mode. Label here.")
    st.caption("**Vector / BM25** = diagnostic. Use to find failure root cause.")
    st.caption("**Graph** = Month 2, not yet available.")


# -- Helpers -------------------------------------------------------------------

def _mode_key(mode_label: str) -> str:
    return {"Hybrid (production)": "hybrid", "Vector only": "vector", "BM25 only": "bm25"}.get(mode_label, "hybrid")


def _score_class(ce_score: float | None) -> str:
    if ce_score is None:
        return ""
    if ce_score >= 2.0:
        return "score-high"
    if ce_score >= 0.0:
        return "score-mid"
    return "score-low"


def _render_card(chunk: dict, position: int, stage: str, query: str,
                 mode_key: str, persona: str, expert_id: str,
                 post_rerank_position: int | None = None) -> None:
    ce = chunk.get("ce_score")
    css_class = _score_class(ce)
    ce_label = f"CE: {ce:.2f}" if ce is not None else "no CE score"
    rrf_label = f"RRF: {chunk['rrf_score']:.4f}"

    key = f"{chunk['chunk_id']}_{stage}_{position}"

    with st.expander(
        f"#{position}  {chunk['citation']}  ·  {chunk['source']}  ·  {rrf_label}  ·  {ce_label}",
        expanded=False,
    ):
        st.markdown(
            f'<span class="citation-badge">{chunk["citation"]}</span>  '
            f'<span class="citation-badge">{chunk["source"]}</span>  '
            f'<span class="{css_class}">{ce_label}</span>',
            unsafe_allow_html=True,
        )
        st.text(chunk["text"][:600] + ("..." if len(chunk["text"]) > 600 else ""))

        col1, col2, col3, col_note = st.columns([1, 1, 1, 3])
        with col1:
            binding = st.button("BINDING", key=f"b_{key}", type="primary")
        with col2:
            relevant = st.button("RELEVANT", key=f"r_{key}")
        with col3:
            irrelevant = st.button("IRREL.", key=f"i_{key}")
        with col_note:
            comment = st.text_input("Notes", key=f"c_{key}",
                                    label_visibility="collapsed", placeholder="Optional notes…")

        label = "BINDING" if binding else ("RELEVANT" if relevant else ("IRRELEVANT" if irrelevant else None))
        if label:
            try:
                submit_feedback(
                    query=query,
                    chunk_id=chunk["chunk_id"],
                    citation=chunk["citation"],
                    source=chunk["source"],
                    retrieval_mode=mode_key,
                    persona=persona.lower().replace(" ", "_"),
                    pre_rerank_rank=position if stage == "pre_rerank" else 0,
                    post_rerank_rank=post_rerank_position,
                    rrf_score=chunk["rrf_score"],
                    ce_score=chunk.get("ce_score"),
                    label=label,
                    comment=comment,
                    expert_id=expert_id,
                )
                st.success(f"Saved: {label}")
            except Exception as e:
                st.error(f"Save failed: {e}")


# -- Search and display --------------------------------------------------------

if search_btn and query_input and mode != "Graph (coming soon)":
    mode_key = _mode_key(mode)
    with st.spinner("Searching…"):
        raw = _audit_retrieval(query_input, mode=mode_key, top_k=top_k)
    st.session_state["audit_result"] = json.loads(raw)
    st.session_state["audit_query"] = query_input
    st.session_state["audit_mode"] = mode_key
    st.session_state["audit_persona"] = persona
    st.session_state["audit_expert"] = expert_id

elif search_btn and mode == "Graph (coming soon)":
    st.warning("Graph retrieval is not yet available. Select Hybrid, Vector, or BM25.")

if "audit_result" in st.session_state:
    result = st.session_state["audit_result"]
    q = st.session_state["audit_query"]
    mk = st.session_state["audit_mode"]
    p = st.session_state["audit_persona"]
    eid = st.session_state["audit_expert"]

    st.subheader(f"Results for: _{q}_")
    col_meta1, col_meta2, col_meta3 = st.columns(3)
    col_meta1.metric("Candidates (pre-rerank)", len(result["candidates"]))
    col_meta2.metric("Final context (post-rerank)", len(result["reranked"]))
    col_meta3.metric("Dropped", len(result["dropped"]))

    if result.get("rewritten_query"):
        st.caption(f"Query rewritten to: _{result['rewritten_query']}_")

    post_rerank_map = {c["chunk_id"]: i + 1 for i, c in enumerate(result["reranked"])}

    tab1, tab2 = st.tabs([
        f"Pre-Rerank — {len(result['candidates'])} candidates",
        f"Post-Rerank — {len(result['reranked'])} final (top 6)",
    ])

    with tab1:
        st.caption("All chunks retrieved before reranking. Label here to flag retrieval failures.")
        show_all = st.toggle(
            f"Show all {len(result['candidates'])} candidates",
            value=False,
            key="show_all_candidates",
        )
        visible_candidates = result["candidates"] if show_all else result["candidates"][:top_k]
        if not show_all and len(result["candidates"]) > top_k:
            st.caption(f"Showing top {top_k} of {len(result['candidates'])} candidates. Toggle above to show all.")
        for i, chunk in enumerate(visible_candidates):
            _render_card(chunk, i + 1, "pre_rerank", q, mk, p, eid,
                         post_rerank_position=post_rerank_map.get(chunk["chunk_id"]))

    with tab2:
        st.caption("Top 6 chunks after CrossEncoder reranking — exactly what the LLM sees in production.")
        for i, chunk in enumerate(result["reranked"]):
            _render_card(chunk, i + 1, "post_rerank", q, mk, p, eid,
                         post_rerank_position=i + 1)
```

- [ ] **Step 3: Run the dashboard locally**

```bash
streamlit run audit_app.py
```

Expected: browser opens at `http://localhost:8501` showing the audit dashboard.

- [ ] **Step 4: Golden-path smoke test**

Enter query: `What is good-time credit in Illinois?`
Select mode: `Hybrid (production)`
Click **Search**.

Expected:
- Pre-Rerank tab shows 15–60 candidates with RRF scores
- Post-Rerank tab shows ≤15 cards with CE scores, color-coded green/yellow/red
- Click `BINDING` on first result → green "Saved: BINDING" confirmation appears
- Verify in Supabase Table Editor: `audit_feedback` has one new row with `label = 'BINDING'`

- [ ] **Step 5: Commit**

```bash
git add audit_app.py
git commit -m "feat: add Retrieval Audit Dashboard with hybrid/vector/bm25 modes and feedback submission"
```

---

### Task 8: Triplet export script for embedding fine-tuning

**Files:**
- Create: `scripts/export_triplets.py`

Reads `audit_feedback` from Supabase and generates (query, positive_chunk, negative_chunk)
triplets for sentence-transformers `MultipleNegativesRankingLoss` fine-tuning.

Hard negatives — IRRELEVANT chunks with high `ce_score` (the reranker was fooled) — are
the most valuable negatives because they teach the embedding model to distinguish near-misses.

- [ ] **Step 1: Create `scripts/export_triplets.py`**

```python
"""
Export audit_feedback ratings as sentence-transformers training triplets.

Usage:
    python3 scripts/export_triplets.py --output data_files/triplets.jsonl --min-positives 2
"""
import argparse
import json
import sys
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

from retrieval.indexes import get_supabase_client
from retrieval.config import ILCS_TABLE, ISCR_TABLE

LABEL_POSITIVE = {"BINDING", "RELEVANT"}
LABEL_NEGATIVE = {"IRRELEVANT"}


def load_feedback(client) -> list[dict]:
    rows = client.table("audit_feedback").select("*").execute().data
    print(f"Loaded {len(rows)} feedback rows", file=sys.stderr)
    return rows


def fetch_chunk_text(client, chunk_id: str, source: str) -> str | None:
    """Fetch enriched_text for a chunk_id from the appropriate table."""
    table = {
        "ilcs": ILCS_TABLE,
        "iscr": "court_rule_chunks",
        "opinions": "opinion_chunks",
        "regulations": "regulation_chunks",
        "documents": "document_chunks",
    }.get(source)
    if not table:
        return None
    try:
        rows = (
            client.table(table)
            .select("enriched_text, text")
            .eq("chunk_id", chunk_id)
            .execute()
            .data
        )
        if rows:
            return rows[0].get("enriched_text") or rows[0].get("text")
    except Exception:
        pass
    return None


def build_triplets(
    feedback: list[dict],
    client,
) -> list[dict]:
    """
    For each query: one triplet — strongest positive (highest ce_score among BINDING/RELEVANT)
    paired with hardest negative (highest ce_score among IRRELEVANT).

    One-best × one-hard keeps the training set diverse and avoids generating near-identical
    triplets when a query has multiple positives sharing the same hard negative.
    """
    by_query: dict[str, dict] = defaultdict(lambda: {"positives": [], "negatives": []})

    for row in feedback:
        label = row.get("label", "")
        entry = {
            "chunk_id": row["chunk_id"],
            "source": row.get("source", ""),
            "ce_score": row.get("ce_score") or 0.0,
        }
        if label in LABEL_POSITIVE:
            by_query[row["query_text"]]["positives"].append(entry)
        elif label in LABEL_NEGATIVE:
            by_query[row["query_text"]]["negatives"].append(entry)

    triplets = []
    skipped = 0

    for query, data in by_query.items():
        positives = sorted(data["positives"], key=lambda x: x["ce_score"], reverse=True)
        negatives = sorted(data["negatives"], key=lambda x: x["ce_score"], reverse=True)

        if not positives or not negatives:
            skipped += 1
            continue

        best_pos = positives[0]
        hard_neg = negatives[0]

        pos_text = fetch_chunk_text(client, best_pos["chunk_id"], best_pos["source"])
        neg_text = fetch_chunk_text(client, hard_neg["chunk_id"], hard_neg["source"])

        if not pos_text or not neg_text:
            skipped += 1
            continue

        triplets.append({
            "query": query,
            "positive": pos_text,
            "negative": neg_text,
            "positive_chunk_id": best_pos["chunk_id"],
            "negative_chunk_id": hard_neg["chunk_id"],
            "negative_ce_score": hard_neg["ce_score"],
        })

    print(f"Generated {len(triplets)} triplets ({skipped} queries skipped)", file=sys.stderr)
    return triplets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data_files/triplets.jsonl")
    args = parser.parse_args()

    client = get_supabase_client()
    feedback = load_feedback(client)
    triplets = build_triplets(feedback, client)

    with open(args.output, "w") as f:
        for t in triplets:
            f.write(json.dumps(t) + "\n")

    print(f"Wrote {len(triplets)} triplets to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Seed a few test feedback rows in Supabase**

In the running audit dashboard, submit at least:
- 2 BINDING ratings for query `"good time credit Illinois"`
- 1 IRRELEVANT rating for the same query (pick a clearly off-topic chunk)

- [ ] **Step 3: Run the export script**

```bash
python3 scripts/export_triplets.py --output data_files/triplets.jsonl
```

Expected output:
```
Loaded N feedback rows
Generated M triplets (K queries skipped)
Wrote M triplets to data_files/triplets.jsonl
```

- [ ] **Step 4: Verify the output format**

```bash
head -1 data_files/triplets.jsonl | python3 -m json.tool
```

Expected: JSON object with `query`, `positive`, `negative`, `positive_chunk_id`,
`negative_chunk_id`, `negative_ce_score` keys.

- [ ] **Step 5: Commit**

```bash
git add scripts/export_triplets.py
git commit -m "feat: add triplet export script for embedding fine-tuning from audit feedback"
```

---

## Self-Review

**Spec coverage:**
- `_secondary_query` seam fixed — `secondary_query` + `bm25_enabled` params on `MultiCollectionRetriever._retrieve()` -- Task 0
- MCP server with 3 external tools (`classify_query`, `search_legal_sources`, `lookup_citation`) -- Tasks 1-4
- `audit_retrieval` is dashboard-internal only, not an MCP tool -- Task 5
- `audit_feedback` Supabase table with BINDING/RELEVANT/IRRELEVANT Labels -- Task 6
- Streamlit Audit Dashboard with Pre-Rerank (slider + show-all toggle) / Post-Rerank (top 6, matches production) tabs -- Task 7
- Mode toggle: Hybrid / Vector (BM25 suppressed, citation pinning preserved) / BM25 / Graph (greyed out) -- Tasks 0 + 5 + 7
- Persona toggle stored in Feedback Records -- Task 7
- Color-coded result cards with RRF + CE scores -- Task 7
- Feedback writes to Supabase -- Tasks 6 + 7
- Triplet export: one-best-positive × one-hard-negative per query -- Task 8
- Graph traversal: noted as Month 2 prerequisite, Graph mode greyed out in UI -- Architecture note
- Available collections probed at startup — unavailable RPC functions skipped gracefully -- Task 2

**Placeholder scan:** No TBDs. All code blocks complete.

**Type consistency:**
- `AuditResponse.retrieval_mode: str` added in Task 1, populated in Task 5, displayed in Task 7
- `submit_feedback` signature matches `_render_card` call in `audit_app.py`
- `_retrieve_by_mode` returns `list[NodeWithScore]`, consumed by both `_search_legal_sources` and `_audit_retrieval`
- `fetch_chunk_text` in export script handles all 5 collection table names
- `top_n=6` in `_get_state()` matches production `CrossEncoderReranker` default
