from unittest.mock import MagicMock, patch
import pytest
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode


def _make_mock_retriever(chunk_id: str, capture_list: list | None = None):
    """Mock FusionRetriever returning one node. Optionally captures _secondary_query at call time."""
    r = MagicMock()
    node = TextNode(id_=chunk_id, text=f"content for {chunk_id}")
    r._secondary_query = None

    def _retrieve(bundle):
        if capture_list is not None:
            capture_list.append(r._secondary_query)
        return [NodeWithScore(node=node, score=0.5)]

    r._retrieve = _retrieve
    return r


def test_multi_collection_retriever_merges_all_collections():
    from retrieval.indexes import MultiCollectionRetriever
    retrievers = [_make_mock_retriever(f"chunk_{i}") for i in range(5)]
    multi = MultiCollectionRetriever(retrievers=retrievers)

    results = multi._retrieve(QueryBundle(query_str="test query"))
    result_ids = {n.node.node_id for n in results}

    assert all(f"chunk_{i}" in result_ids for i in range(5))


def test_secondary_query_propagated_to_all_retrievers():
    from retrieval.indexes import MultiCollectionRetriever
    captured = []
    retrievers = [_make_mock_retriever(f"chunk_{i}", capture_list=captured) for i in range(3)]
    multi = MultiCollectionRetriever(retrievers=retrievers)
    multi._secondary_query = "rewritten query"

    multi._retrieve(QueryBundle(query_str="original"))

    assert all(q == "rewritten query" for q in captured)


def test_secondary_query_cleared_after_retrieve():
    from retrieval.indexes import MultiCollectionRetriever
    r = _make_mock_retriever("chunk_0")
    multi = MultiCollectionRetriever(retrievers=[r])
    multi._secondary_query = "rewritten query"
    multi._retrieve(QueryBundle(query_str="original"))

    assert r._secondary_query is None


def test_secondary_query_cleared_on_exception():
    from retrieval.indexes import MultiCollectionRetriever
    r = MagicMock()
    r._secondary_query = None
    r._retrieve = MagicMock(side_effect=RuntimeError("simulated error"))

    multi = MultiCollectionRetriever(retrievers=[r])
    multi._secondary_query = "rewritten query"

    with pytest.raises(RuntimeError):
        multi._retrieve(QueryBundle(query_str="original"))

    assert r._secondary_query is None


def test_build_all_retrievers_creates_one_per_collection():
    from retrieval.indexes import build_all_retrievers, FusionRetriever
    from retrieval.config import COLLECTIONS

    with patch("retrieval.indexes.build_fusion_retriever") as mock_build:
        mock_build.return_value = MagicMock(spec=FusionRetriever)
        retrievers = build_all_retrievers(MagicMock(), MagicMock())

    assert len(retrievers) == len(COLLECTIONS)
    assert mock_build.call_count == len(COLLECTIONS)
    # Each call uses a different RPC from the registry
    called_rpcs = [call.args[2] for call in mock_build.call_args_list]
    assert called_rpcs == [c.rpc for c in COLLECTIONS]
