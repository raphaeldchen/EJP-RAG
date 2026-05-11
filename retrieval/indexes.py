import re
from supabase import Client, create_client
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from retrieval.config import (
    SUPABASE_URL,
    SUPABASE_SERVICE_KEY,
    ILCS_TABLE,
    ISCR_TABLE,
    DEFAULT_TOP_K,
)
from retrieval.embeddings import get_embedding_model
from retrieval.vector_store import SupabaseRPCVectorStore
from retrieval.bm25_store import BM25Retriever

_ILCS_CITATION_RE = re.compile(r'\d+\s+ILCS\s+\d+/[\d\.\-]+')
_RULE_RE = re.compile(r'\bRule\s+(\d+)\b', re.IGNORECASE)


def get_supabase_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


class FusionRetriever(BaseRetriever):
    """
    Pure vector retriever for a single collection. BM25 and citation pinning are
    handled at the MultiCollectionRetriever level.
    """
    def __init__(
        self,
        vector_index: VectorStoreIndex,
        top_k: int = DEFAULT_TOP_K,
    ):
        self._vector_retriever = vector_index.as_retriever(similarity_top_k=top_k)
        self._top_k = top_k
        super().__init__()

    def _retrieve(
        self,
        query_bundle: QueryBundle,
        secondary_query: str | None = None,
    ) -> list[NodeWithScore]:
        primary = self._vector_retriever.retrieve(query_bundle)

        if secondary_query:
            from retrieval.postprocessor import merge_ranked_lists
            sec_bundle = QueryBundle(query_str=secondary_query)
            secondary = self._vector_retriever.retrieve(sec_bundle)
            return merge_ranked_lists([primary, secondary], top_n=40, weights=[1.0, 0.5])

        return primary


def build_fusion_retriever(
    client: Client,
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
    return FusionRetriever(vector_index=index, top_k=top_k)


def build_all_retrievers(
    client: Client,
) -> dict[str, FusionRetriever]:
    from retrieval.config import COLLECTIONS
    return {
        col.id: build_fusion_retriever(client, col.rpc)
        for col in COLLECTIONS
    }


# Per-collection RRF weight multipliers. "bm25" gives the single shared BM25 arm
# one authoritative vote without the previous 5× inflation from per-FusionRetriever BM25.
_DEFAULT_COLLECTION_WEIGHTS: dict[str, float] = {
    "ilcs":        1.5,
    "iscr":        1.5,
    "opinions":    0.8,
    "regulations": 1.0,
    "documents":   0.7,
    "bm25":        1.2,
}


class MultiCollectionRetriever(BaseRetriever):
    """
    Runs one FusionRetriever (pure vector) per collection plus a single shared BM25 arm.
    Merges all arms via RRF so BM25 counts as one vote regardless of collection count.
    Citation pinning runs once here instead of redundantly in each sub-retriever.
    """

    def __init__(
        self,
        retrievers: list[FusionRetriever],
        bm25: BM25Retriever,
        client: Client,
        collection_ids: list[str] | None = None,
        weights: dict[str, float] | None = None,
    ):
        self._retrievers = retrievers
        self._bm25 = bm25
        self._client = client
        self._collection_ids = collection_ids or []
        self._weights = weights or {}
        super().__init__()

    def retrieve(
        self,
        str_or_query_bundle,
        secondary_query: str | None = None,
        bm25_enabled: bool = True,
    ) -> list[NodeWithScore]:
        from llama_index.core.schema import QueryBundle as QB
        if isinstance(str_or_query_bundle, str):
            query_bundle = QB(query_str=str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle
        return self._retrieve(query_bundle, secondary_query=secondary_query, bm25_enabled=bm25_enabled)

    def _retrieve(
        self,
        query_bundle: QueryBundle,
        secondary_query: str | None = None,
        bm25_enabled: bool = True,
    ) -> list[NodeWithScore]:
        from retrieval.postprocessor import merge_ranked_lists

        query_str = query_bundle.query_str
        results = [r._retrieve(query_bundle, secondary_query=secondary_query) for r in self._retrievers]

        per_list_weights = (
            [self._weights.get(cid, 1.0) for cid in self._collection_ids]
            if self._weights and self._collection_ids
            else [1.0] * len(results)
        )

        all_lists = results
        all_weights = per_list_weights

        if bm25_enabled:
            # Single BM25 arm — one vote regardless of how many collections are searched
            bm25_weight = self._weights.get("bm25", 1.2)
            bm25_nodes = self._bm25.retrieve(query_str, top_k=DEFAULT_TOP_K)
            bm25_list = [NodeWithScore(node=n, score=0.0) for n in bm25_nodes]
            all_lists = results + [bm25_list]
            all_weights = per_list_weights + [bm25_weight]

            # Secondary BM25 arm at half weight (matches secondary vector fusion weight)
            if secondary_query:
                sec_bm25_nodes = self._bm25.retrieve(secondary_query, top_k=DEFAULT_TOP_K)
                sec_bm25_list = [NodeWithScore(node=n, score=0.0) for n in sec_bm25_nodes]
                all_lists.append(sec_bm25_list)
                all_weights.append(bm25_weight * 0.5)

        fused = merge_ranked_lists(all_lists, top_n=60, weights=all_weights)

        # Citation pinning — always on regardless of bm25_enabled
        combined = query_str + (" " + secondary_query if secondary_query else "")
        existing_ids = {n.node.node_id for n in fused}
        pinned: list[NodeWithScore] = []

        ilcs_citations = _ILCS_CITATION_RE.findall(combined)
        if ilcs_citations:
            ilcs_pinned = self._fetch_by_citation(ilcs_citations, exclude_ids=existing_ids)
            if ilcs_pinned:
                print(f"[Citation] Pinned {len(ilcs_pinned)} ILCS chunk(s) for: {ilcs_citations}")
            pinned.extend(ilcs_pinned)

        rule_numbers = _RULE_RE.findall(combined)
        if rule_numbers:
            rule_pinned = self._fetch_by_rule_number(rule_numbers, exclude_ids=existing_ids)
            if rule_pinned:
                print(f"[Citation] Pinned {len(rule_pinned)} ISCR chunk(s) for: Rule {rule_numbers}")
            pinned.extend(rule_pinned)

        if pinned:
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
                nodes.append(NodeWithScore(node=node, score=1.0))
                exclude_ids.add(row["chunk_id"])
        return nodes

    def _fetch_by_rule_number(
        self, rule_numbers: list[str], exclude_ids: set[str]
    ) -> list[NodeWithScore]:
        nodes = []
        for rule_number in rule_numbers:
            try:
                rows = (
                    self._client.table(ISCR_TABLE)
                    .select("chunk_id, enriched_text, text, rule_number, rule_title")
                    .eq("rule_number", rule_number.strip())
                    .execute()
                    .data
                )
            except Exception as e:
                print(f"[Citation] Rule lookup failed for Rule {rule_number!r}: {e}")
                continue
            for row in rows:
                if row["chunk_id"] in exclude_ids:
                    continue
                node = TextNode(
                    id_=row["chunk_id"],
                    text=row.get("enriched_text") or row.get("text", ""),
                    metadata={
                        "rule_number": row.get("rule_number"),
                        "rule_title": row.get("rule_title"),
                        "pinned": True,
                    },
                )
                nodes.append(NodeWithScore(node=node, score=1.0))
                exclude_ids.add(row["chunk_id"])
        return nodes


def build_multi_retriever(
    client: Client,
    bm25: BM25Retriever,
    retrievers: dict[str, FusionRetriever] | None = None,
    weights: dict[str, float] | None = _DEFAULT_COLLECTION_WEIGHTS,
) -> MultiCollectionRetriever:
    if retrievers is None:
        retrievers = build_all_retrievers(client)
    return MultiCollectionRetriever(
        retrievers=list(retrievers.values()),
        bm25=bm25,
        client=client,
        collection_ids=list(retrievers.keys()),
        weights=weights,
    )
