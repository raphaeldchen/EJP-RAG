from dataclasses import dataclass
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from sentence_transformers import CrossEncoder


_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def merge_ranked_lists(
    ranked_lists: list[list[NodeWithScore]],
    k: int = 60,
    top_n: int = 40,
) -> list[NodeWithScore]:
    """
    RRF over N pre-ranked NodeWithScore lists (e.g. results from multiple queries).
    Each list is treated as an independent ranking signal.
    """
    scores: dict[str, float] = {}
    node_map: dict[str, NodeWithScore] = {}

    for ranked in ranked_lists:
        for rank, node_with_score in enumerate(ranked):
            chunk_id = node_with_score.node.node_id
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (rank + k)
            if chunk_id not in node_map:
                node_map[chunk_id] = node_with_score

    ranked_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [
        NodeWithScore(node=node_map[cid].node, score=s)
        for cid, s in ranked_ids
        if cid in node_map
    ]


def reciprocal_rank_fusion(
    vector_nodes: list[NodeWithScore],
    bm25_nodes: list[TextNode],
    k: int = 60,
    top_n: int = 40,
) -> list[NodeWithScore]:
    """
    Merge two ranked lists using RRF.
    k=60 is the standard constant — dampens the impact of very high ranks.
    """
    scores: dict[str, float] = {}
    node_map: dict[str, NodeWithScore] = {}

    # Score from vector results
    for rank, node_with_score in enumerate(vector_nodes):
        chunk_id = node_with_score.node.node_id
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (rank + k)
        node_map[chunk_id] = node_with_score

    # Score from BM25 results
    for rank, node in enumerate(bm25_nodes):
        chunk_id = node.node_id
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (rank + k)
        if chunk_id not in node_map:
            # BM25-only hit — wrap in NodeWithScore
            node_map[chunk_id] = NodeWithScore(node=node, score=0.0)

    # Sort by fused score, take top_n
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return [
        NodeWithScore(
            node=node_map[chunk_id].node,
            score=rrf_score,
        )
        for chunk_id, rrf_score in ranked
        if chunk_id in node_map
    ]


class CrossEncoderReranker(BaseNodePostprocessor):
    """
    Reranks nodes using a cross-encoder model.
    Drops nodes below score_threshold.
    """
    model_name: str = _CROSS_ENCODER_MODEL
    top_n: int = 6
    score_threshold: float = -3.0  # ms-marco scores legal text in roughly -10..+10; 0.1 was too aggressive
    _model: CrossEncoder | None = None

    class Config:
        arbitrary_types_allowed = True

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            print(f"[Reranker] Loading {self.model_name}...")
            self._model = CrossEncoder(self.model_name)
        return self._model

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        if not nodes or query_bundle is None:
            return nodes

        model = self._get_model()
        query = query_bundle.query_str
        pairs = [(query, n.node.get_content()) for n in nodes]
        scores = model.predict(pairs)

        # DEBUG — remove after tuning
        print(f"\n[Reranker] scores for '{query[:60]}':")
        for n, s in sorted(zip(nodes, scores), key=lambda x: x[1], reverse=True):
            preview = n.node.get_content()[:60].replace("\n", " ")
            print(f"  {s:.3f} | {preview}")

        reranked = [
            NodeWithScore(node=n.node, score=float(s))
            for n, s in zip(nodes, scores)
            if float(s) >= self.score_threshold
        ]
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:self.top_n]