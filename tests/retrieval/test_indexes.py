from unittest.mock import MagicMock, call, patch
import pytest
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode


def _make_mock_retriever(chunk_id: str, capture_list: list | None = None):
    """Mock FusionRetriever returning one node. Optionally captures secondary_query at call time."""
    r = MagicMock()
    node = TextNode(id_=chunk_id, text=f"content for {chunk_id}")

    def _retrieve(bundle, secondary_query=None):
        if capture_list is not None:
            capture_list.append(secondary_query)
        return [NodeWithScore(node=node, score=0.5)]

    r._retrieve = _retrieve
    return r


def _make_mock_bm25():
    bm25 = MagicMock()
    bm25.retrieve.return_value = []
    return bm25


def test_multi_collection_retriever_merges_all_collections():
    from retrieval.indexes import MultiCollectionRetriever
    retrievers = [_make_mock_retriever(f"chunk_{i}") for i in range(5)]
    multi = MultiCollectionRetriever(retrievers=retrievers, bm25=_make_mock_bm25(), client=MagicMock())

    results = multi._retrieve(QueryBundle(query_str="test query"))
    result_ids = {n.node.node_id for n in results}

    assert all(f"chunk_{i}" in result_ids for i in range(5))


def test_secondary_query_propagated_to_all_retrievers():
    from retrieval.indexes import MultiCollectionRetriever
    captured = []
    retrievers = [_make_mock_retriever(f"chunk_{i}", capture_list=captured) for i in range(3)]
    multi = MultiCollectionRetriever(retrievers=retrievers, bm25=_make_mock_bm25(), client=MagicMock())

    multi._retrieve(QueryBundle(query_str="original"), secondary_query="rewritten query")

    assert all(q == "rewritten query" for q in captured)


def test_retrieve_override_passes_secondary_query():
    from retrieval.indexes import MultiCollectionRetriever
    captured = []
    retrievers = [_make_mock_retriever("chunk_0", capture_list=captured)]
    multi = MultiCollectionRetriever(retrievers=retrievers, bm25=_make_mock_bm25(), client=MagicMock())

    multi.retrieve("original query", secondary_query="rewritten query")

    assert captured == ["rewritten query"]


def test_bm25_disabled_skips_bm25_arm():
    from retrieval.indexes import MultiCollectionRetriever
    bm25 = _make_mock_bm25()
    multi = MultiCollectionRetriever(
        retrievers=[_make_mock_retriever("chunk_0")],
        bm25=bm25,
        client=MagicMock(),
    )

    multi._retrieve(QueryBundle(query_str="test"), bm25_enabled=False)

    bm25.retrieve.assert_not_called()


def test_secondary_query_propagates_on_exception():
    from retrieval.indexes import MultiCollectionRetriever
    r = MagicMock()
    r._retrieve = MagicMock(side_effect=RuntimeError("simulated error"))

    multi = MultiCollectionRetriever(retrievers=[r], bm25=_make_mock_bm25(), client=MagicMock())

    with pytest.raises(RuntimeError):
        multi._retrieve(QueryBundle(query_str="original"), secondary_query="rewritten query")

    r._retrieve.assert_called_once_with(
        QueryBundle(query_str="original"), secondary_query="rewritten query"
    )


def test_build_all_retrievers_creates_one_per_collection():
    from retrieval.indexes import build_all_retrievers, FusionRetriever
    from retrieval.config import COLLECTIONS

    with patch("retrieval.indexes.build_fusion_retriever") as mock_build:
        mock_build.return_value = MagicMock(spec=FusionRetriever)
        retrievers = build_all_retrievers(MagicMock())

    assert len(retrievers) == len(COLLECTIONS)
    assert mock_build.call_count == len(COLLECTIONS)
    # rpc_function is now args[1] (client, rpc_function, top_k)
    called_rpcs = [call.args[1] for call in mock_build.call_args_list]
    assert called_rpcs == [c.rpc for c in COLLECTIONS]
