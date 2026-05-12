import hashlib as _hashlib
import re as _re
from dataclasses import dataclass

from mcp.server.fastmcp import FastMCP
from llama_index.core import Settings
from llama_index.core.schema import NodeWithScore, QueryBundle

from retrieval.config import ILCS_TABLE, ISCR_TABLE, DEFAULT_TOP_K
from retrieval.embeddings import get_embedding_model
from retrieval.indexes import (
    get_supabase_client,
    build_fusion_retriever,
    build_multi_retriever,
    MultiCollectionRetriever,
)
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


def _probe_collections(client) -> dict[str, object]:
    """Build only collections whose RPC functions are registered in Supabase."""
    from retrieval.config import COLLECTIONS
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
    if mode == "bm25":
        bm25_nodes = state.retriever._bm25.retrieve(query_str, top_k=DEFAULT_TOP_K)
        if secondary_query:
            sec_nodes = state.retriever._bm25.retrieve(secondary_query, top_k=DEFAULT_TOP_K)
            seen = {n.node_id for n in bm25_nodes}
            bm25_nodes = bm25_nodes + [n for n in sec_nodes if n.node_id not in seen]
        return [NodeWithScore(node=n, score=0.0) for n in bm25_nodes]

    if mode == "vector":
        return state.retriever.retrieve(query_str, secondary_query=secondary_query, bm25_enabled=False)

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


def _audit_retrieval(query: str, mode: str = "hybrid", top_k: int = 20) -> str:
    state = _get_state()
    reflection = reflect(query)
    candidates = _retrieve_by_mode(state, query, reflection.rewritten_query, mode=mode)

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


def submit_feedback(
    query: str,
    chunk_id: str,
    citation: str,
    source: str,
    retrieval_mode: str,
    pre_rerank_rank: int,
    post_rerank_rank: int | None,
    rrf_score: float,
    ce_score: float | None,
    label: str,
    comment: str = "",
    expert_id: str = "",
    persona: str = "",
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


if __name__ == "__main__":
    mcp.run()
