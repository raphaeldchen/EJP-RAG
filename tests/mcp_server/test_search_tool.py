import json
from mcp_server.server import _search_legal_sources


def test_search_returns_chunk_results():
    result = json.loads(_search_legal_sources("good time credit Illinois sentencing"))
    assert result["intent"] == "in_scope"
    assert len(result["results"]) > 0
    first = result["results"][0]
    assert first["chunk_id"] and first["text"] and first["citation"]
    assert "source" in first  # field always present; BM25 nodes may return "unknown"


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
