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
    r.bm25.index(corpus_tokens, show_progress=False)
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
    # bm25s.BM25().index([]) raises ValueError; set bm25 to None to signal empty.
    r.bm25 = None
    results = r.retrieve("any query", top_k=5)
    assert results == []
