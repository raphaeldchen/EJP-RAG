import re
from supabase import Client, create_client
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from retrieval.config import (
    SUPABASE_URL,
    SUPABASE_SERVICE_KEY,
    ILCS_TABLE,
    ILCS_RPC,
    DEFAULT_TOP_K,
)
from retrieval.embeddings import get_embedding_model
from retrieval.vector_store import SupabaseRPCVectorStore
from retrieval.bm25_store import BM25Retriever

_ILCS_CITATION_RE = re.compile(r'\d+\s+ILCS\s+\d+/[\d\.\-]+')


def get_supabase_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


class FusionRetriever(BaseRetriever):
    """
    Runs vector search and BM25 in parallel, then injects any chunks that
    are explicitly cited in the query by section_citation (citation pinning).

    Set _secondary_query before calling retrieve() to enable multi-query mode:
    retrieval runs for both queries independently and results are merged via RRF.
    Intended usage: primary = original natural-language query (better semantic
    embedding), secondary = reflection-rewritten query (citation keyword signals).
    Citation pinning scans both queries so ILCS citations in the rewrite still
    get injected regardless of which is primary.
    """
    def __init__(
        self,
        vector_index: VectorStoreIndex,
        bm25: BM25Retriever,
        supabase_client: Client,
        top_k: int = DEFAULT_TOP_K,
    ):
        self._vector_retriever = vector_index.as_retriever(
            similarity_top_k=top_k
        )
        self._bm25 = bm25
        self._client = supabase_client
        self._top_k = top_k
        self._secondary_query: str | None = None
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        from retrieval.postprocessor import reciprocal_rank_fusion, merge_ranked_lists

        query_str = query_bundle.query_str

        # Primary retrieval
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        bm25_nodes = self._bm25.retrieve(query_str, top_k=self._top_k)
        primary_fused = reciprocal_rank_fusion(
            vector_nodes=vector_nodes,
            bm25_nodes=bm25_nodes,
            top_n=40,
        )

        # Secondary retrieval (multi-query: original query alongside rewritten)
        if self._secondary_query:
            sec_bundle = QueryBundle(query_str=self._secondary_query)
            sec_vector = self._vector_retriever.retrieve(sec_bundle)
            sec_bm25 = self._bm25.retrieve(self._secondary_query, top_k=self._top_k)
            secondary_fused = reciprocal_rank_fusion(
                vector_nodes=sec_vector,
                bm25_nodes=sec_bm25,
                top_n=40,
            )
            fused = merge_ranked_lists(
                [primary_fused, secondary_fused],
                top_n=40,
                weights=[1.0, 0.5],
            )
        else:
            fused = primary_fused

        # Citation pinning: inject chunks for any ILCS sections explicitly named in either
        # the primary or secondary query. Citations typically appear in the rewritten query,
        # but scanning both ensures nothing is missed regardless of which is primary.
        combined = query_str + (" " + self._secondary_query if self._secondary_query else "")
        citations = _ILCS_CITATION_RE.findall(combined)
        if citations:
            existing_ids = {n.node.node_id for n in fused}
            pinned = self._fetch_by_citation(citations, exclude_ids=existing_ids)
            if pinned:
                print(f"[Citation] Pinned {len(pinned)} chunk(s) for: {citations}")
            fused = pinned + fused

        return fused

    def _fetch_by_citation(
        self, citations: list[str], exclude_ids: set[str]
    ) -> list[NodeWithScore]:
        nodes = []
        for citation in citations:
            try:
                rows = (
                    self._client.table(ILCS_TABLE)
                    .select("chunk_id, enriched_text, text, section_citation, major_topic")
                    .eq("section_citation", citation.strip())
                    .execute()
                    .data
                )
            except Exception as e:
                print(f"[Citation] Lookup failed for {citation!r}: {e}")
                continue
            for row in rows:
                if row["chunk_id"] in exclude_ids:
                    continue
                node = TextNode(
                    id_=row["chunk_id"],
                    text=row.get("enriched_text") or row.get("text", ""),
                    metadata={
                        "section_citation": row.get("section_citation"),
                        "major_topic": row.get("major_topic"),
                        "pinned": True,
                    },
                )
                # Give pinned nodes a higher score than any RRF result (max RRF ≈ 1/60 ≈ 0.017)
                nodes.append(NodeWithScore(node=node, score=1.0))
                exclude_ids.add(row["chunk_id"])
        return nodes


def build_fusion_retriever(
    client: Client,
    bm25: BM25Retriever,
    rpc_function: str,
    top_k: int = DEFAULT_TOP_K,
) -> FusionRetriever:
    store = SupabaseRPCVectorStore(
        supabase_client=client,
        rpc_function=rpc_function,
    )
    index = VectorStoreIndex.from_vector_store(
        vector_store=store,
        embed_model=get_embedding_model(),
    )
    return FusionRetriever(
        vector_index=index,
        bm25=bm25,
        supabase_client=client,
        top_k=top_k,
    )


def build_all_retrievers(
    client: Client,
    bm25: BM25Retriever,
) -> list[FusionRetriever]:
    from retrieval.config import COLLECTIONS
    return [
        build_fusion_retriever(client, bm25, col.rpc)
        for col in COLLECTIONS
    ]


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


def build_multi_retriever(
    client: Client,
    bm25: BM25Retriever,
    retrievers: list[FusionRetriever] | None = None,
) -> MultiCollectionRetriever:
    if retrievers is None:
        retrievers = build_all_retrievers(client, bm25)
    return MultiCollectionRetriever(retrievers=retrievers)