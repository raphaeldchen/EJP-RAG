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
