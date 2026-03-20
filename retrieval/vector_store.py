from typing import Any, Optional
from pydantic import BaseModel, ConfigDict
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)

class SupabaseRPCVectorStore(BasePydanticVectorStore):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    supabase_client: Any
    rpc_function: str
    text_field: str = "enriched_text"
    stores_text: bool = True
    flat_metadata: bool = False
    rpc_filters: dict = {}

    @property
    def client(self) -> Any:
        return self.supabase_client

    def add(self, nodes, **kwargs):
        raise NotImplementedError("Use the embed pipeline to insert nodes.")

    def delete(self, ref_doc_id: str, **kwargs):
        raise NotImplementedError("Deletion not implemented.")

    def query(
        self, query: VectorStoreQuery, **kwargs
    ) -> VectorStoreQueryResult:
        embedding = query.query_embedding
        top_k = query.similarity_top_k or 10

        params = {
            "query_embedding": embedding,
            "match_count": top_k,
            **self.rpc_filters,
        }

        response = self.supabase_client.rpc(self.rpc_function, params).execute()

        if not response.data:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        nodes = []
        similarities = []
        ids = []

        for row in response.data:
            metadata = {
                k: v
                for k, v in row.items()
                if k not in ("text", "enriched_text", "similarity", "chunk_id")
            }
            if "metadata" in metadata and isinstance(metadata["metadata"], dict):
                metadata.update(metadata.pop("metadata"))
            node = TextNode(
                id_=row["chunk_id"],
                text=row.get(self.text_field) or row.get("text", ""),
                metadata=metadata,
            )
            nodes.append(node)
            similarities.append(row["similarity"])
            ids.append(row["chunk_id"])
        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids,
        )