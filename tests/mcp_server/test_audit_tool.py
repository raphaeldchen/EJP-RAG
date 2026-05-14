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


def test_submit_feedback_deduplicates_by_query_chunk_expert():
    """Second call for same (query, chunk, expert) must update, not insert."""
    from mcp_server.server import submit_feedback, _get_state

    shared = dict(
        query="good time credit",
        chunk_id="test-chunk-dedup",
        citation="730 ILCS 5/3-6-3",
        source="ilcs",
        retrieval_mode="hybrid",
        persona="researcher",
        pre_rerank_rank=1,
        post_rerank_rank=1,
        rrf_score=0.05,
        ce_score=2.1,
        expert_id="test-expert",
    )

    state = _get_state()
    try:
        # First submission
        submit_feedback(**shared, label="RELEVANT", comment="first")
        # Second submission — different label, same triple
        submit_feedback(**shared, label="BINDING", comment="changed my mind")

        rows = (
            state.client.table("audit_feedback")
            .select("label, comment")
            .eq("chunk_id", "test-chunk-dedup")
            .eq("expert_id", "test-expert")
            .execute()
            .data
        )
        assert len(rows) == 1, f"Expected 1 row, got {len(rows)} — dedup not working"
        assert rows[0]["label"] == "BINDING", "Latest label should have won"
        assert rows[0]["comment"] == "changed my mind"
    finally:
        state.client.table("audit_feedback").delete().eq("chunk_id", "test-chunk-dedup").eq("expert_id", "test-expert").execute()


def test_submit_feedback_anonymous_normalizes_expert_id():
    """Blank expert_id should be stored as 'anonymous', enabling dedup for unauthenticated users."""
    from mcp_server.server import submit_feedback, _get_state

    state = _get_state()
    try:
        submit_feedback(
            query="good time credit anon",
            chunk_id="test-chunk-anon",
            citation="730 ILCS 5/3-6-3",
            source="ilcs",
            retrieval_mode="hybrid",
            persona="",
            pre_rerank_rank=1,
            post_rerank_rank=None,
            rrf_score=0.05,
            ce_score=None,
            label="RELEVANT",
            comment="",
            expert_id="",   # blank — should normalize to "anonymous"
        )
        # Second submission — same triple, different label — proves normalization enables dedup
        submit_feedback(
            query="good time credit anon",
            chunk_id="test-chunk-anon",
            citation="730 ILCS 5/3-6-3",
            source="ilcs",
            retrieval_mode="hybrid",
            persona="",
            pre_rerank_rank=1,
            post_rerank_rank=None,
            rrf_score=0.05,
            ce_score=None,
            label="BINDING",
            comment="",
            expert_id="",   # blank again — still "anonymous", same triple
        )

        rows = (
            state.client.table("audit_feedback")
            .select("expert_id")
            .eq("chunk_id", "test-chunk-anon")
            .execute()
            .data
        )
        assert rows[0]["expert_id"] == "anonymous"
        assert len(rows) == 1, f"Expected 1 row, got {len(rows)} — dedup not working for anonymous"
    finally:
        state.client.table("audit_feedback").delete().eq("chunk_id", "test-chunk-anon").execute()


def test_get_feedback_history_returns_expert_rows():
    """get_feedback_history returns only rows for the given expert_id."""
    from mcp_server.server import submit_feedback, get_feedback_history, _get_state

    state = _get_state()
    try:
        submit_feedback(
            query="test history query",
            chunk_id="test-chunk-history",
            citation="730 ILCS 5/3-6-3",
            source="ilcs",
            retrieval_mode="hybrid",
            persona="",
            pre_rerank_rank=1,
            post_rerank_rank=1,
            rrf_score=0.05,
            ce_score=2.1,
            label="BINDING",
            comment="test note",
            expert_id="test-history-expert",
        )

        rows = get_feedback_history("test-history-expert")
        our_row = next((r for r in rows if r["chunk_id"] == "test-chunk-history"), None)
        assert our_row is not None, "Row not returned for correct expert_id"
        assert our_row["label"] == "BINDING"
        assert our_row["query_text"] == "test history query"
        assert our_row["citation"] == "730 ILCS 5/3-6-3"
        assert our_row["comment"] == "test note"
        assert our_row["post_rerank_rank"] == 1
        assert "query_id" in our_row
        assert "created_at" in our_row

        other_rows = get_feedback_history("different-expert")
        assert not any(r["chunk_id"] == "test-chunk-history" for r in other_rows), \
            "Row leaked to a different expert_id"
    finally:
        state.client.table("audit_feedback").delete() \
            .eq("chunk_id", "test-chunk-history") \
            .eq("expert_id", "test-history-expert") \
            .execute()
