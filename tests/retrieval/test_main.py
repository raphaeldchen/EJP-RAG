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
