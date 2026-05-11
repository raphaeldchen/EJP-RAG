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
