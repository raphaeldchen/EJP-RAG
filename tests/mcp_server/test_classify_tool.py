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
