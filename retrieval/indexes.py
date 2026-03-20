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


def get_supabase_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


class FusionRetriever(BaseRetriever):
    """
    Runs vector search and BM25 in parallel,
    returns combined results for the postprocessor to rerank.
    """
    def __init__(
        self,
        vector_index: VectorStoreIndex,
        bm25: BM25Retriever,
        top_k: int = DEFAULT_TOP_K,
    ):
        self._vector_retriever = vector_index.as_retriever(
            similarity_top_k=top_k
        )
        self._bm25 = bm25
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
        return fused


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
    return FusionRetriever(vector_index=index, bm25=bm25, top_k=top_k)


def build_all_retrievers(
    client: Client,
    bm25: BM25Retriever,
) -> dict[str, FusionRetriever]:
    return {
        "ilcs": build_fusion_retriever(client, bm25, ILCS_RPC),
        "iscr": build_fusion_retriever(client, bm25, ISCR_RPC),
    }