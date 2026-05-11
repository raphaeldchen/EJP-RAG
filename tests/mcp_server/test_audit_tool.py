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
    assert len(vector["candidates"]) <= len(hybrid["candidates"]) + 5


def test_audit_bm25_mode_returns_candidates():
    result = json.loads(_audit_retrieval("good time credit", mode="bm25"))
    assert result["retrieval_mode"] == "bm25"
    assert len(result["candidates"]) > 0


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
