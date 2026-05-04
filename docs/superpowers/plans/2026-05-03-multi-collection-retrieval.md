# Multi-Collection Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the retrieval system from 2 hardcoded sources (ILCS + ISCR) to 5 collections (opinions, regulations, documents) using a `CollectionConfig` registry, bm25s disk-cached BM25, and a `MultiCollectionRetriever`.

**Architecture:** A `CollectionConfig` dataclass in `config.py` drives iteration across all files — `bm25_store.py` and `indexes.py` iterate `COLLECTIONS` instead of hardcoding table names. BM25 is rebuilt from all 5 Supabase tables and cached to disk via `bm25s` (numpy-backed, memory-mapped loading) so startups after the initial build are near-instant. `DualFusionRetriever` is renamed `MultiCollectionRetriever` and holds a `list[FusionRetriever]` instead of named pairs. Corpus data (chunk IDs, texts, metadata) is cached as JSON alongside the bm25s index files.

**Tech Stack:** `bm25s` (new), `supabase-py`, `llama-index`, `pytest`

**Spec:** `docs/superpowers/specs/2026-05-03-multi-collection-retrieval-design.md`

---

## File Map

| Action | Path | Responsibility |
|--------|------|---------------|
| Modify | `retrieval/config.py` | Add `CollectionConfig` dataclass + `COLLECTIONS` registry |
| Modify | `retrieval/bm25_store.py` | Swap `rank_bm25` → `bm25s`; iterate registry; add disk cache |
| Create | `retrieval/bm25_build.py` | Standalone CLI to force BM25 index rebuild |
| Modify | `retrieval/indexes.py` | Rename `DualFusionRetriever` → `MultiCollectionRetriever`; list-based arms |
| Modify | `retrieval/reflection.py` | Expand system prompt to describe all 5 collection types |
| Modify | `retrieval/main.py` | Unify citation extraction via `display_citation`; add 3 new test queries |
| Modify | `retrieval/query_engine.py` | Update type hint only |
| Create | `tests/retrieval/__init__.py` | Empty package marker |
| Create | `tests/retrieval/test_config.py` | CollectionConfig + registry tests |
| Create | `tests/retrieval/test_bm25_store.py` | BM25 cache + retrieval unit tests |
| Create | `tests/retrieval/test_indexes.py` | MultiCollectionRetriever tests |
| Create | `tests/retrieval/test_main.py` | Citation extraction + reflection prompt tests |

---

## Task 1: Install bm25s

**Files:**
- No code files change

- [ ] **Step 1: Install bm25s into the project venv**

```bash
pip install bm25s
```

Expected output: `Successfully installed bm25s-...`

- [ ] **Step 2: Verify import and basic API**

```bash
python3 -c "import bm25s; r = bm25s.BM25(); r.index([['hello', 'world'], ['foo', 'bar']]); print('bm25s OK')"
```

Expected: `bm25s OK`

- [ ] **Step 3: Commit**

```bash
git add -u
git commit -m "chore: install bm25s for disk-cached BM25 index"
```

---

## Task 2: CollectionConfig registry in config.py

**Files:**
- Modify: `retrieval/config.py`
- Create: `tests/retrieval/__init__.py`
- Create: `tests/retrieval/test_config.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/retrieval/__init__.py` (empty file — no content needed).

Create `tests/retrieval/test_config.py`:

```python
def test_collections_has_five_entries():
    from retrieval.config import COLLECTIONS
    assert len(COLLECTIONS) == 5


def test_collection_ids():
    from retrieval.config import COLLECTIONS
    ids = [c.id for c in COLLECTIONS]
    assert ids == ["ilcs", "iscr", "opinions", "regulations", "documents"]


def test_collection_fields_all_non_empty():
    from retrieval.config import COLLECTIONS
    for col in COLLECTIONS:
        assert col.id
        assert col.table
        assert col.rpc


def test_ilcs_entry_reads_from_env_var(monkeypatch):
    monkeypatch.setenv("ILCS_TABLE", "ilcs_chunks_test")
    monkeypatch.setenv("ILCS_RPC", "match_ilcs_chunks_test")
    import importlib
    import retrieval.config as cfg_module
    importlib.reload(cfg_module)
    from retrieval.config import COLLECTIONS
    ilcs = next(c for c in COLLECTIONS if c.id == "ilcs")
    assert ilcs.table == "ilcs_chunks_test"
    assert ilcs.rpc == "match_ilcs_chunks_test"
    importlib.reload(cfg_module)


def test_opinion_chunks_table():
    from retrieval.config import COLLECTIONS
    opinions = next(c for c in COLLECTIONS if c.id == "opinions")
    assert opinions.table == "opinion_chunks"
    assert opinions.rpc == "match_opinion_chunks"


def test_regulation_chunks_table():
    from retrieval.config import COLLECTIONS
    regulations = next(c for c in COLLECTIONS if c.id == "regulations")
    assert regulations.table == "regulation_chunks"
    assert regulations.rpc == "match_regulation_chunks"


def test_document_chunks_table():
    from retrieval.config import COLLECTIONS
    documents = next(c for c in COLLECTIONS if c.id == "documents")
    assert documents.table == "document_chunks"
    assert documents.rpc == "match_document_chunks"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/retrieval/test_config.py -v
```

Expected: `ImportError` or `AttributeError` — `CollectionConfig` not yet defined.

- [ ] **Step 3: Add CollectionConfig and COLLECTIONS to config.py**

Open `retrieval/config.py`. Add at the top of the file, after `import os`:

```python
from dataclasses import dataclass
```

Add after the existing `ISCR_RPC` line at the bottom of the file:

```python

@dataclass(frozen=True)
class CollectionConfig:
    id: str    # "ilcs" | "iscr" | "opinions" | "regulations" | "documents"
    table: str # Supabase table name
    rpc: str   # vector search RPC function name


COLLECTIONS: list[CollectionConfig] = [
    CollectionConfig("ilcs",        ILCS_TABLE,             ILCS_RPC),
    CollectionConfig("iscr",        ISCR_TABLE,             ISCR_RPC),
    CollectionConfig("opinions",    "opinion_chunks",        "match_opinion_chunks"),
    CollectionConfig("regulations", "regulation_chunks",     "match_regulation_chunks"),
    CollectionConfig("documents",   "document_chunks",       "match_document_chunks"),
]
```

Note: `ILCS_TABLE`, `ISCR_TABLE`, `ILCS_RPC`, `ISCR_RPC` are the existing module-level constants that read from env vars. The `ilcs` and `iscr` registry entries inherit those values so experiment-table overrides still work.

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/retrieval/test_config.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add retrieval/config.py tests/retrieval/__init__.py tests/retrieval/test_config.py
git commit -m "feat: add CollectionConfig registry to config.py"
```

---

## Task 3: BM25Retriever — bm25s + disk cache

**Files:**
- Modify: `retrieval/bm25_store.py`
- Create: `tests/retrieval/test_bm25_store.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/retrieval/test_bm25_store.py`:

```python
import json
from pathlib import Path
from unittest.mock import MagicMock

import bm25s
import pytest


# ---------------------------------------------------------------------------
# _fetch_counts
# ---------------------------------------------------------------------------

def test_fetch_counts_returns_dict_with_collection_ids():
    from retrieval.bm25_store import _fetch_counts
    from retrieval.config import COLLECTIONS

    client = MagicMock()
    execute_result = MagicMock()
    execute_result.count = 42
    (client.table.return_value
           .select.return_value
           .limit.return_value
           .execute.return_value) = execute_result

    counts = _fetch_counts(client)

    assert set(counts.keys()) == {c.id for c in COLLECTIONS}
    assert all(v == 42 for v in counts.values())


# ---------------------------------------------------------------------------
# _cache_is_fresh
# ---------------------------------------------------------------------------

def test_cache_is_fresh_missing_meta(tmp_path, monkeypatch):
    import retrieval.bm25_store as bs
    monkeypatch.setattr(bs, "_BM25_META_PATH", tmp_path / "meta.json")
    monkeypatch.setattr(bs, "_BM25_INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(bs, "_BM25_CORPUS_PATH", tmp_path / "corpus.json")
    assert bs._cache_is_fresh({"ilcs": 10}) is False


def test_cache_is_fresh_missing_index_dir(tmp_path, monkeypatch):
    import retrieval.bm25_store as bs
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"ilcs": 10}))
    monkeypatch.setattr(bs, "_BM25_META_PATH", meta)
    monkeypatch.setattr(bs, "_BM25_INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(bs, "_BM25_CORPUS_PATH", tmp_path / "corpus.json")
    assert bs._cache_is_fresh({"ilcs": 10}) is False


def test_cache_is_fresh_missing_corpus(tmp_path, monkeypatch):
    import retrieval.bm25_store as bs
    meta = tmp_path / "meta.json"
    index_dir = tmp_path / "index"
    meta.write_text(json.dumps({"ilcs": 10}))
    index_dir.mkdir()
    monkeypatch.setattr(bs, "_BM25_META_PATH", meta)
    monkeypatch.setattr(bs, "_BM25_INDEX_DIR", index_dir)
    monkeypatch.setattr(bs, "_BM25_CORPUS_PATH", tmp_path / "corpus.json")
    assert bs._cache_is_fresh({"ilcs": 10}) is False


def test_cache_is_fresh_counts_match(tmp_path, monkeypatch):
    import retrieval.bm25_store as bs
    meta = tmp_path / "meta.json"
    index_dir = tmp_path / "index"
    corpus = tmp_path / "corpus.json"
    counts = {"ilcs": 100, "iscr": 50, "opinions": 337000, "regulations": 3000, "documents": 7000}
    meta.write_text(json.dumps(counts))
    index_dir.mkdir()
    corpus.write_text("{}")
    monkeypatch.setattr(bs, "_BM25_META_PATH", meta)
    monkeypatch.setattr(bs, "_BM25_INDEX_DIR", index_dir)
    monkeypatch.setattr(bs, "_BM25_CORPUS_PATH", corpus)
    assert bs._cache_is_fresh(counts) is True


def test_cache_is_fresh_counts_differ(tmp_path, monkeypatch):
    import retrieval.bm25_store as bs
    meta = tmp_path / "meta.json"
    index_dir = tmp_path / "index"
    corpus = tmp_path / "corpus.json"
    stored = {"ilcs": 100, "iscr": 50, "opinions": 337000, "regulations": 3000, "documents": 7000}
    current = {"ilcs": 101, "iscr": 50, "opinions": 337000, "regulations": 3000, "documents": 7000}
    meta.write_text(json.dumps(stored))
    index_dir.mkdir()
    corpus.write_text("{}")
    monkeypatch.setattr(bs, "_BM25_META_PATH", meta)
    monkeypatch.setattr(bs, "_BM25_INDEX_DIR", index_dir)
    monkeypatch.setattr(bs, "_BM25_CORPUS_PATH", corpus)
    assert bs._cache_is_fresh(current) is False


# ---------------------------------------------------------------------------
# BM25Retriever.retrieve()
# ---------------------------------------------------------------------------

def _make_retriever(texts: list[str], enriched: list[str], chunk_ids: list[str]):
    """Build a BM25Retriever with state set directly — no Supabase needed."""
    from retrieval.bm25_store import BM25Retriever, _tokenize
    r = BM25Retriever.__new__(BM25Retriever)
    r.chunk_ids = chunk_ids
    r.texts = texts
    r.enriched_texts = enriched
    r._metadata = [{} for _ in texts]
    corpus_tokens = [_tokenize(t) for t in texts]
    r.bm25 = bm25s.BM25()
    r.bm25.index(corpus_tokens)
    return r


def test_retrieve_returns_text_nodes():
    from llama_index.core.schema import TextNode
    r = _make_retriever(
        texts=["right to counsel defendant", "sentencing guidelines class 1 felony"],
        enriched=["[ILCS] right to counsel", "[ILCS] sentencing"],
        chunk_ids=["c1", "c2"],
    )
    results = r.retrieve("right to counsel", top_k=2)
    assert len(results) >= 1
    assert all(isinstance(n, TextNode) for n in results)


def test_retrieve_top_result_is_most_relevant():
    r = _make_retriever(
        texts=["right to counsel defendant", "sentencing guidelines class 1 felony", "parole IDOC"],
        enriched=["[ILCS] counsel", "[ILCS] sentencing", "[IAC] parole"],
        chunk_ids=["c1", "c2", "c3"],
    )
    results = r.retrieve("right to counsel defendant", top_k=3)
    assert results[0].node_id == "c1"


def test_retrieve_uses_enriched_text_for_node_body():
    r = _make_retriever(
        texts=["sentencing guidelines"],
        enriched=["[730 ILCS 5/5-4.5]: sentencing guidelines"],
        chunk_ids=["c1"],
    )
    results = r.retrieve("sentencing guidelines", top_k=1)
    assert results[0].text == "[730 ILCS 5/5-4.5]: sentencing guidelines"


def test_retrieve_metadata_includes_bm25_score():
    r = _make_retriever(
        texts=["right to counsel"],
        enriched=["[ILCS] right to counsel"],
        chunk_ids=["c1"],
    )
    results = r.retrieve("right to counsel", top_k=1)
    assert "bm25_score" in results[0].metadata
    assert results[0].metadata["bm25_score"] > 0


def test_retrieve_empty_corpus():
    from retrieval.bm25_store import BM25Retriever
    r = BM25Retriever.__new__(BM25Retriever)
    r.chunk_ids = []
    r.texts = []
    r.enriched_texts = []
    r._metadata = []
    r.bm25 = bm25s.BM25()
    r.bm25.index([])
    results = r.retrieve("any query", top_k=5)
    assert results == []
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/retrieval/test_bm25_store.py -v
```

Expected: `ImportError` — `_fetch_counts`, `_cache_is_fresh` not yet defined.

- [ ] **Step 3: Rewrite retrieval/bm25_store.py**

Replace the entire file contents with:

```python
import json
import re
from pathlib import Path

import bm25s
from supabase import Client
from llama_index.core.schema import TextNode

from retrieval.config import COLLECTIONS

BM25_CACHE_DIR = Path("data_files/bm25_cache")
_BM25_INDEX_DIR = BM25_CACHE_DIR / "index"
_BM25_CORPUS_PATH = BM25_CACHE_DIR / "corpus.json"
_BM25_META_PATH = BM25_CACHE_DIR / "meta.json"


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    # Preserve statute citations like 5/7-1, 12-3.05 before stripping punctuation
    statute_pattern = re.findall(r'\d+/\d+[\-\.\d]*', text)
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    tokens.extend(statute_pattern)
    return tokens


def _fetch_counts(client: Client) -> dict[str, int]:
    """Return row counts for each collection — used as a cache staleness watermark."""
    counts = {}
    for col in COLLECTIONS:
        result = (
            client.table(col.table)
            .select("chunk_id", count="exact")
            .limit(0)
            .execute()
        )
        counts[col.id] = result.count or 0
    return counts


def _cache_is_fresh(current_counts: dict[str, int]) -> bool:
    if not _BM25_META_PATH.exists():
        return False
    if not _BM25_INDEX_DIR.exists():
        return False
    if not _BM25_CORPUS_PATH.exists():
        return False
    stored = json.loads(_BM25_META_PATH.read_text())
    return stored == current_counts


class BM25Retriever:
    def __init__(self, client: Client):
        self.chunk_ids: list[str] = []
        self.texts: list[str] = []
        self.enriched_texts: list[str] = []
        self._metadata: list[dict] = []
        self.bm25: bm25s.BM25 | None = None
        self._load(client)

    def _load(self, client: Client):
        print("[BM25] Checking cache...")
        current_counts = _fetch_counts(client)

        if _cache_is_fresh(current_counts):
            print("[BM25] Cache is fresh — loading from disk (mmap)...")
            self._load_from_cache()
        else:
            print("[BM25] Cache stale or missing — rebuilding from Supabase...")
            self._build_from_supabase(client)
            self._save_cache(current_counts)

        print(f"[BM25] Index ready: {len(self.chunk_ids)} chunks")

    def _load_from_cache(self):
        self.bm25 = bm25s.BM25.load(str(_BM25_INDEX_DIR), mmap=True)
        corpus = json.loads(_BM25_CORPUS_PATH.read_text(encoding="utf-8"))
        self.chunk_ids = corpus["chunk_ids"]
        self.texts = corpus["texts"]
        self.enriched_texts = corpus["enriched_texts"]
        self._metadata = corpus["metadata"]

    def _build_from_supabase(self, client: Client):
        def fetch_all(table: str) -> list[dict]:
            rows = []
            page_size = 1000
            offset = 0
            while True:
                batch = (
                    client.table(table)
                    .select("chunk_id, text, enriched_text, display_citation")
                    .not_.is_("text", "null")
                    .range(offset, offset + page_size - 1)
                    .execute()
                    .data
                )
                rows.extend(batch)
                if len(batch) < page_size:
                    break
                offset += page_size
            return rows

        all_rows: list[dict] = []
        for col in COLLECTIONS:
            print(f"[BM25] Fetching {col.table}...")
            all_rows.extend(fetch_all(col.table))

        self.chunk_ids = [r["chunk_id"] for r in all_rows]
        self.texts = [r["text"] for r in all_rows]
        self.enriched_texts = [r.get("enriched_text") or r["text"] for r in all_rows]
        self._metadata = [
            {k: v for k, v in r.items()
             if k not in ("chunk_id", "text", "enriched_text") and v is not None}
            for r in all_rows
        ]

        print(f"[BM25] Tokenizing {len(self.texts)} chunks...")
        corpus_tokens = [_tokenize(t) for t in self.texts]
        self.bm25 = bm25s.BM25()
        self.bm25.index(corpus_tokens)

    def _save_cache(self, counts: dict[str, int]):
        print("[BM25] Saving cache to disk...")
        BM25_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _BM25_INDEX_DIR.mkdir(parents=True, exist_ok=True)

        self.bm25.save(str(_BM25_INDEX_DIR))
        _BM25_CORPUS_PATH.write_text(
            json.dumps({
                "chunk_ids": self.chunk_ids,
                "texts": self.texts,
                "enriched_texts": self.enriched_texts,
                "metadata": self._metadata,
            }, ensure_ascii=False),
            encoding="utf-8",
        )
        _BM25_META_PATH.write_text(json.dumps(counts))
        print("[BM25] Cache saved.")

    def retrieve(self, query: str, top_k: int = 20) -> list[TextNode]:
        query_tokens = [_tokenize(query)]
        k = min(top_k, len(self.chunk_ids))
        if k == 0:
            return []

        results, scores = self.bm25.retrieve(query_tokens, k=k)

        nodes = []
        for idx, score in zip(results[0], scores[0]):
            idx = int(idx)
            score = float(score)
            if score <= 0:
                continue
            metadata = {"bm25_score": score, **self._metadata[idx]}
            node = TextNode(
                id_=self.chunk_ids[idx],
                text=self.enriched_texts[idx],
                metadata=metadata,
            )
            nodes.append(node)

        return nodes
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/retrieval/test_bm25_store.py -v
```

Expected: `11 passed`

- [ ] **Step 5: Commit**

```bash
git add retrieval/bm25_store.py tests/retrieval/test_bm25_store.py
git commit -m "feat: rewrite BM25Retriever with bm25s + disk cache across all 5 collections"
```

---

## Task 4: bm25_build.py standalone rebuild script

**Files:**
- Create: `retrieval/bm25_build.py`

- [ ] **Step 1: Create retrieval/bm25_build.py**

```python
"""Force a full BM25 index rebuild from Supabase.

Usage: python3 -m retrieval.bm25_build

Deletes the existing cache and rebuilds from scratch. Run this after
a batch_embed job completes to keep the index current without waiting
for a count-change at the next startup.
"""
import shutil
from dotenv import load_dotenv
from supabase import create_client

from retrieval.bm25_store import BM25_CACHE_DIR, BM25Retriever
from retrieval.config import SUPABASE_URL, SUPABASE_SERVICE_KEY


def main():
    load_dotenv()
    if BM25_CACHE_DIR.exists():
        shutil.rmtree(BM25_CACHE_DIR)
        print(f"[bm25_build] Cleared {BM25_CACHE_DIR}")
    client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    BM25Retriever(client)
    print("[bm25_build] Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it imports cleanly**

```bash
python3 -c "from retrieval.bm25_build import main; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add retrieval/bm25_build.py
git commit -m "feat: add bm25_build.py standalone index rebuild script"
```

---

## Task 5: MultiCollectionRetriever in indexes.py

**Files:**
- Modify: `retrieval/indexes.py`
- Create: `tests/retrieval/test_indexes.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/retrieval/test_indexes.py`:

```python
from unittest.mock import MagicMock, patch
import pytest
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode


def _make_mock_retriever(chunk_id: str, capture_list: list | None = None):
    """Mock FusionRetriever returning one node. Optionally captures _secondary_query at call time."""
    r = MagicMock()
    node = TextNode(id_=chunk_id, text=f"content for {chunk_id}")
    r._secondary_query = None

    def _retrieve(bundle):
        if capture_list is not None:
            capture_list.append(r._secondary_query)
        return [NodeWithScore(node=node, score=0.5)]

    r._retrieve = _retrieve
    return r


def test_multi_collection_retriever_merges_all_collections():
    from retrieval.indexes import MultiCollectionRetriever
    retrievers = [_make_mock_retriever(f"chunk_{i}") for i in range(5)]
    multi = MultiCollectionRetriever(retrievers=retrievers)

    results = multi._retrieve(QueryBundle(query_str="test query"))
    result_ids = {n.node.node_id for n in results}

    assert all(f"chunk_{i}" in result_ids for i in range(5))


def test_secondary_query_propagated_to_all_retrievers():
    from retrieval.indexes import MultiCollectionRetriever
    captured = []
    retrievers = [_make_mock_retriever(f"chunk_{i}", capture_list=captured) for i in range(3)]
    multi = MultiCollectionRetriever(retrievers=retrievers)
    multi._secondary_query = "rewritten query"

    multi._retrieve(QueryBundle(query_str="original"))

    assert all(q == "rewritten query" for q in captured)


def test_secondary_query_cleared_after_retrieve():
    from retrieval.indexes import MultiCollectionRetriever
    r = _make_mock_retriever("chunk_0")
    multi = MultiCollectionRetriever(retrievers=[r])
    multi._secondary_query = "rewritten query"
    multi._retrieve(QueryBundle(query_str="original"))

    assert r._secondary_query is None


def test_secondary_query_cleared_on_exception():
    from retrieval.indexes import MultiCollectionRetriever
    r = MagicMock()
    r._secondary_query = None
    r._retrieve = MagicMock(side_effect=RuntimeError("simulated error"))

    multi = MultiCollectionRetriever(retrievers=[r])
    multi._secondary_query = "rewritten query"

    with pytest.raises(RuntimeError):
        multi._retrieve(QueryBundle(query_str="original"))

    assert r._secondary_query is None


def test_build_all_retrievers_creates_one_per_collection():
    from retrieval.indexes import build_all_retrievers, FusionRetriever
    from retrieval.config import COLLECTIONS

    with patch("retrieval.indexes.build_fusion_retriever") as mock_build:
        mock_build.return_value = MagicMock(spec=FusionRetriever)
        retrievers = build_all_retrievers(MagicMock(), MagicMock())

    assert len(retrievers) == len(COLLECTIONS)
    assert mock_build.call_count == len(COLLECTIONS)
    # Each call uses a different RPC from the registry
    called_rpcs = [call.args[2] for call in mock_build.call_args_list]
    assert called_rpcs == [c.rpc for c in COLLECTIONS]
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/retrieval/test_indexes.py -v
```

Expected: `ImportError` — `MultiCollectionRetriever` not yet defined.

- [ ] **Step 3: Make three targeted edits to retrieval/indexes.py**

**Edit 1** — Replace the `DualFusionRetriever` class entirely with `MultiCollectionRetriever`:

```python
class MultiCollectionRetriever(BaseRetriever):
    """
    Runs one FusionRetriever per collection in parallel and merges results
    via RRF. Handles cross-domain queries without a router — all collections
    are always searched and the CrossEncoder reranker is the final arbiter.

    Set _secondary_query before calling retrieve() to enable multi-query mode;
    it is propagated to all sub-retrievers and cleared in a finally block.
    """

    def __init__(self, retrievers: list[FusionRetriever]):
        self._retrievers = retrievers
        self._secondary_query: str | None = None
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        from retrieval.postprocessor import merge_ranked_lists

        for r in self._retrievers:
            r._secondary_query = self._secondary_query
        try:
            results = [r._retrieve(query_bundle) for r in self._retrievers]
        finally:
            for r in self._retrievers:
                r._secondary_query = None

        return merge_ranked_lists(results, top_n=40)
```

**Edit 2** — Replace `build_all_retrievers` with the registry-iterating version:

```python
def build_all_retrievers(
    client: Client,
    bm25: BM25Retriever,
) -> list[FusionRetriever]:
    from retrieval.config import COLLECTIONS
    return [
        build_fusion_retriever(client, bm25, col.rpc)
        for col in COLLECTIONS
    ]
```

**Edit 3** — Replace `build_dual_retriever` with `build_multi_retriever`:

```python
def build_multi_retriever(
    client: Client,
    bm25: BM25Retriever,
    retrievers: list[FusionRetriever] | None = None,
) -> MultiCollectionRetriever:
    if retrievers is None:
        retrievers = build_all_retrievers(client, bm25)
    return MultiCollectionRetriever(retrievers=retrievers)
```

Also update the `from retrieval.config import ...` line at the top of `indexes.py`. `ISCR_RPC` is no longer needed (the registry handles it); `ILCS_TABLE` and `ILCS_RPC` remain because `_fetch_by_citation` still uses them directly:

```python
from retrieval.config import (
    SUPABASE_URL,
    SUPABASE_SERVICE_KEY,
    ILCS_TABLE,
    ILCS_RPC,
    DEFAULT_TOP_K,
)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/retrieval/test_indexes.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add retrieval/indexes.py tests/retrieval/test_indexes.py
git commit -m "feat: replace DualFusionRetriever with MultiCollectionRetriever over 5 collections"
```

---

## Task 6: Expand reflection.py system prompt

**Files:**
- Modify: `retrieval/reflection.py`
- Create: `tests/retrieval/test_main.py` (initial version)

- [ ] **Step 1: Write the failing tests**

Create `tests/retrieval/test_main.py`:

```python
def test_reflection_prompt_describes_court_opinions():
    from retrieval.reflection import _SYSTEM_PROMPT
    assert "Court opinions" in _SYSTEM_PROMPT
    assert "7th Circuit" in _SYSTEM_PROMPT
    assert "1973" in _SYSTEM_PROMPT


def test_reflection_prompt_describes_regulations():
    from retrieval.reflection import _SYSTEM_PROMPT
    assert "IDOC" in _SYSTEM_PROMPT
    assert "Administrative" in _SYSTEM_PROMPT


def test_reflection_prompt_describes_policy_docs():
    from retrieval.reflection import _SYSTEM_PROMPT
    assert "SPAC" in _SYSTEM_PROMPT
    assert "ICCB" in _SYSTEM_PROMPT
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/retrieval/test_main.py -v
```

Expected: `3 failed`

- [ ] **Step 3: Replace `_SYSTEM_PROMPT` in retrieval/reflection.py**

Replace the entire `_SYSTEM_PROMPT` string (from the `_SYSTEM_PROMPT = """` line to its closing `"""`) with:

```python
_SYSTEM_PROMPT = """You are a query processor for a legal research system covering Illinois criminal justice law.

The corpus contains:
- ILCS (Illinois Compiled Statutes):
  - 720 ILCS — Criminal Offenses (elements of crimes, defenses, definitions)
  - 725 ILCS — Criminal Procedure (arrest, bail, trial, sentencing procedures)
  - 730 ILCS — Corrections and Sentencing (sentencing ranges, good-time credit, parole, probation)
  - 705 ILCS — Courts, including juvenile justice (705 ILCS 405)
  - 625 ILCS — Vehicles, including DUI offenses
  - 430 ILCS — Fire Safety, including FOID Card Act (430 ILCS 65) and Concealed Carry Act (430 ILCS 66)
  - 750 ILCS — Family, including Domestic Violence Act and orders of protection
  - 775 ILCS — Civil Rights, including rights of crime victims and defendants
  - 735 ILCS — Civil Procedure, including post-conviction relief and habeas corpus
  - 410 ILCS — Public Health, including drug treatment programs and sexual assault response
  - 325 ILCS — Employment, including background check restrictions and collateral consequences of conviction
  - 225 ILCS — Professions and Occupations, including licensing consequences of criminal convictions
  - 50 ILCS — Local Government, including county jail administration and sheriff authority
  - 20 ILCS — Executive agency acts with any criminal-justice nexus (broadly construed): Department of Corrections (20 ILCS 1005), Prisoner Review Board (20 ILCS 1405), expungement and sealing (20 ILCS 2630), Alcoholism and Drug Abuse Act (20 ILCS 301), Department of Human Services (20 ILCS 1305), Department on Aging (20 ILCS 105), Illinois Violence Prevention Authority (20 ILCS 1335), Criminal Justice Information Authority (20 ILCS 3930), and many others. The 20 ILCS corpus is broad — when a query references any 20 ILCS chapter, default to in_scope unless it is unmistakably unrelated to criminal justice.
- ISCR (Illinois Supreme Court Rules): procedural court rules covering appeals, filing deadlines, discovery, and jury selection
- Court opinions:
  - Illinois Supreme Court and Appellate Court opinions (1973–2024) via CAP bulk download
  - 7th Circuit federal opinions via CourtListener
  - Use for questions about judicial interpretation, constitutional challenges, sentencing precedent, and how courts have applied specific statutes
- Regulations and directives:
  - Illinois Administrative Code Title 20 (519 IDOC-relevant sections)
  - IDOC Administrative Directives (103 records) and reentry resources
  - Use for questions about IDOC facility rules, disciplinary procedures, programming requirements, and reentry planning
- Policy and advocacy documents:
  - SPAC (Sentencing Policy Advisory Council) publications
  - ICCB correctional education enrollment reports FY2020–2025
  - Federal Register rules, BOP policy, ED Dear Colleague Letters on federal law intersecting Illinois prisoners
  - Restore Justice IL resources
  - Cook County Public Defender resources
  - Use for sentencing policy trends, correctional education data, and advocacy resources

Your job is to classify the query and, when needed, rewrite it into precise statutory language that will retrieve the most relevant chunks from the corpus.

Classify as exactly one of:
- in_scope: query is about Illinois criminal law or Illinois court procedure — use this even for colloquial or vague queries that clearly relate to Illinois criminal topics; always provide a rewritten_query with the relevant ILCS or Rule citation(s) if you know them
- out_of_scope: ONLY use this for queries that are clearly about federal law, another state's law, or a topic with no conceivable connection to Illinois courts or criminal justice. When in doubt between in_scope and ambiguous, prefer in_scope.
- ambiguous: genuinely unclear whether the topic is within scope — use sparingly

Important: ALL Illinois Supreme Court Rules (ISCR) are in scope. Criminal defendants rely on rules governing discovery (Rule 201–214), depositions (Rule 202), affidavits (Rule 191), sanctions (Rule 137), jury selection (Rule 431–434), appeals (Rule 604–610), and interpreter appointment (Rule 46). Do not classify an ISCR query as out_of_scope.

When rewriting:
- Include the most specific ILCS or Rule citation you know based on your knowledge of Illinois law
- Expand colloquial language into precise legal terminology (e.g. "beat a charge" → "affirmative defenses and grounds for suppression")
- Make the rewritten query specific enough to retrieve the right statutory sections
- Provide rewritten_query for ALL in_scope queries, not just ambiguous ones — it improves retrieval
- For multi-statute queries, include ALL relevant citations you know

Respond ONLY with this JSON:
{
  "intent": "in_scope" | "out_of_scope" | "ambiguous",
  "reasoning": "one sentence",
  "rewritten_query": "rewritten query with ILCS citations where known, else null"
}"""
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/retrieval/test_main.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add retrieval/reflection.py tests/retrieval/test_main.py
git commit -m "feat: expand reflection prompt to describe all 5 collection types"
```

---

## Task 7: Citation extraction + test queries in main.py

**Files:**
- Modify: `retrieval/main.py`
- Modify: `tests/retrieval/test_main.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/retrieval/test_main.py`:

```python
from unittest.mock import MagicMock
from llama_index.core.schema import NodeWithScore, TextNode


def _make_response(nodes_metadata: list[dict]) -> MagicMock:
    response = MagicMock()
    response.source_nodes = [
        NodeWithScore(node=TextNode(id_="t", text="c", metadata=m), score=1.0)
        for m in nodes_metadata
    ]
    return response


def test_extract_citation_uses_display_citation():
    from retrieval.main import _extract_citations
    response = _make_response([{"display_citation": "People v. Smith, 2010 IL 109103"}])
    assert _extract_citations(response) == ["People v. Smith, 2010 IL 109103"]


def test_extract_citation_ilcs_fallback():
    from retrieval.main import _extract_citations
    response = _make_response([{"section_citation": "730 ILCS 5/3-6-3"}])
    assert _extract_citations(response) == ["730 ILCS 5/3-6-3"]


def test_extract_citation_iscr_fallback():
    from retrieval.main import _extract_citations
    response = _make_response([{"rule_number": "401", "rule_title": "Rule 401 — Waiver of Counsel"}])
    assert _extract_citations(response) == ["Rule 401 — Waiver of Counsel"]


def test_extract_citation_deduplicates():
    from retrieval.main import _extract_citations
    response = _make_response([
        {"display_citation": "People v. Smith (2010)"},
        {"display_citation": "People v. Smith (2010)"},
    ])
    assert _extract_citations(response) == ["People v. Smith (2010)"]


def test_extract_citation_display_citation_takes_precedence():
    from retrieval.main import _extract_citations
    response = _make_response([{
        "display_citation": "People v. Jones (2015)",
        "section_citation": "720 ILCS 5/9-1",
    }])
    assert _extract_citations(response) == ["People v. Jones (2015)"]


def test_extract_citation_empty_display_citation_falls_through():
    from retrieval.main import _extract_citations
    response = _make_response([{
        "display_citation": "",
        "section_citation": "720 ILCS 5/9-1",
    }])
    assert _extract_citations(response) == ["720 ILCS 5/9-1"]


def test_extract_citation_no_source_nodes():
    from retrieval.main import _extract_citations
    response = MagicMock(spec=[])
    assert _extract_citations(response) == []
```

- [ ] **Step 2: Run to confirm failures**

```bash
pytest tests/retrieval/test_main.py -v -k "extract_citation"
```

Expected: multiple failures — current `_extract_citations` doesn't check `display_citation`.

- [ ] **Step 3: Replace `_extract_citations` in retrieval/main.py**

Replace the existing `_extract_citations` function with:

```python
def _extract_citations(response) -> list[str]:
    citations = []
    seen = set()

    if not hasattr(response, "source_nodes"):
        return citations

    for node_with_score in response.source_nodes:
        meta = node_with_score.node.metadata

        # New sources: display_citation is the canonical field
        dc = (meta.get("display_citation") or "").strip()
        if dc and dc not in seen:
            seen.add(dc)
            citations.append(dc)
            continue

        # Legacy: ilcs_chunks uses section_citation
        section = meta.get("section_citation")
        if section and section not in seen:
            seen.add(section)
            citations.append(section)
            continue

        # Legacy: court_rule_chunks uses rule_number + rule_title
        rule = meta.get("rule_number")
        if rule:
            title = meta.get("rule_title", "")
            if title and title.startswith(f"Rule {rule}"):
                title = title[len(f"Rule {rule}"):].lstrip(" .").strip()
            label = f"Rule {rule}" + (f" — {title}" if title else "")
            if label not in seen:
                seen.add(label)
                citations.append(label)

    return citations
```

- [ ] **Step 4: Update imports and rename in main.py**

Change the import line from:
```python
from retrieval.indexes import get_supabase_client, build_dual_retriever
```
to:
```python
from retrieval.indexes import get_supabase_client, build_multi_retriever
```

In `build_rag()`, change:
```python
dual_retriever = build_dual_retriever(client, bm25)
```
to:
```python
dual_retriever = build_multi_retriever(client, bm25)
```

(Keep the local variable named `dual_retriever` to avoid cascading renames in the same function — or rename throughout if you prefer consistency.)

- [ ] **Step 5: Add 3 new test queries in main.py**

In the `test_queries` list inside `if __name__ == "__main__":`, add:

```python
# --- Court opinions (CAP / CourtListener) ---
"What has the Illinois Supreme Court held about proportionality review for extended-term sentences?",

# --- IDOC regulations ---
"What does IDOC policy say about disciplinary segregation and access to educational programming?",

# --- Policy documents ---
"What does SPAC data show about the racial composition of Illinois prison admissions for drug offenses?",
```

- [ ] **Step 6: Run all main.py tests**

```bash
pytest tests/retrieval/test_main.py -v
```

Expected: `10 passed`

- [ ] **Step 7: Commit**

```bash
git add retrieval/main.py tests/retrieval/test_main.py
git commit -m "feat: unify citation extraction via display_citation; update main.py imports; add 3 new test queries"
```

---

## Task 8: query_engine.py type hint update

**Files:**
- Modify: `retrieval/query_engine.py`

- [ ] **Step 1: Update the import and type hint**

In `retrieval/query_engine.py`, change:

```python
# Old
from retrieval.indexes import DualFusionRetriever
```

```python
# New
from retrieval.indexes import MultiCollectionRetriever
```

Change the function signature:

```python
# Old
def build_query_engine(
    dual_retriever: DualFusionRetriever,
    ...

# New
def build_query_engine(
    dual_retriever: MultiCollectionRetriever,
    ...
```

- [ ] **Step 2: Verify import is clean**

```bash
python3 -c "from retrieval.query_engine import build_query_engine; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add retrieval/query_engine.py
git commit -m "chore: update query_engine.py type hint to MultiCollectionRetriever"
```

---

## Task 9: Full test run + smoke check

- [ ] **Step 1: Run the full retrieval test suite**

```bash
pytest tests/retrieval/ -v
```

Expected: all tests pass.

- [ ] **Step 2: Verify the whole retrieval module imports cleanly**

```bash
python3 -c "
from retrieval.config import COLLECTIONS
from retrieval.bm25_store import BM25Retriever
from retrieval.indexes import MultiCollectionRetriever, build_multi_retriever
from retrieval.reflection import reflect
from retrieval.main import build_rag, query
print('All imports OK')
print('Collections:', [c.id for c in COLLECTIONS])
"
```

Expected:
```
All imports OK
Collections: ['ilcs', 'iscr', 'opinions', 'regulations', 'documents']
```

- [ ] **Step 3: Final commit**

```bash
git add -u
git commit -m "chore: final integration check — multi-collection retrieval complete"
```

---

## Post-implementation

Once all tasks pass, trigger the first BM25 index build (only needed once; subsequent startups load from disk in under a second):

```bash
python3 -m retrieval.bm25_build
```

This takes ~4–5 minutes to fetch all 5 collections from Supabase, tokenize ~350k chunks, and write the cache to `data_files/bm25_cache/`. Every subsequent startup loads from disk.

Then run eval to confirm ranking quality is preserved after the `rank_bm25` → `bm25s` switch:

```bash
python3 -m eval.run_eval
```

Compare nDCG@6 before and after. The bm25s BM25+ variant may differ slightly from rank_bm25's BM25Okapi — the eval delta confirms whether the switch preserved or improved quality.
