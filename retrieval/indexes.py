import re
from supabase import Client, create_client
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from retrieval.config import (
    SUPABASE_URL,
    SUPABASE_SERVICE_KEY,
    ILCS_RPC,
    ISCR_RPC,
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
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        from retrieval.postprocessor import reciprocal_rank_fusion

        query_str = query_bundle.query_str

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        bm25_nodes = self._bm25.retrieve(query_str, top_k=self._top_k)

        fused = reciprocal_rank_fusion(
            vector_nodes=vector_nodes,
            bm25_nodes=bm25_nodes,
            top_n=20,
        )

        # Citation pinning: inject chunks for any ILCS sections explicitly named in the query.
        # This ensures that when reflection rewrites to include e.g. "730 ILCS 5/5-4.5-30",
        # those exact statute chunks always reach the cross-encoder regardless of embedding similarity.
        citations = _ILCS_CITATION_RE.findall(query_str)
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
                    self._client.table("ilcs_chunks")
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
) -> dict[str, FusionRetriever]:
    return {
        "ilcs": build_fusion_retriever(client, bm25, ILCS_RPC),
        "iscr": build_fusion_retriever(client, bm25, ISCR_RPC),
    }