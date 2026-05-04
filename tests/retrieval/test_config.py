def test_collections_has_five_entries():
    from retrieval.config import COLLECTIONS
    assert len(COLLECTIONS) == 5


def test_collection_ids():
    from retrieval.config import COLLECTIONS
    ids = [c.id for c in COLLECTIONS]
    assert ids == ["ilcs", "iscr", "opinions", "regulations", "documents"]


def test_collection_fields_all_non_empty():
    from retrieval.config import COLLECTIONS
    for col in COLLECTIONS:
        assert col.id
        assert col.table
        assert col.rpc


def test_ilcs_entry_reads_from_env_var(monkeypatch):
    monkeypatch.setenv("ILCS_TABLE", "ilcs_chunks_test")
    monkeypatch.setenv("ILCS_RPC", "match_ilcs_chunks_test")
    import importlib
    import retrieval.config as cfg_module
    importlib.reload(cfg_module)
    from retrieval.config import COLLECTIONS
    ilcs = next(c for c in COLLECTIONS if c.id == "ilcs")
    assert ilcs.table == "ilcs_chunks_test"
    assert ilcs.rpc == "match_ilcs_chunks_test"
    importlib.reload(cfg_module)


def test_opinion_chunks_table():
    from retrieval.config import COLLECTIONS
    opinions = next(c for c in COLLECTIONS if c.id == "opinions")
    assert opinions.table == "opinion_chunks"
    assert opinions.rpc == "match_opinion_chunks"


def test_regulation_chunks_table():
    from retrieval.config import COLLECTIONS
    regulations = next(c for c in COLLECTIONS if c.id == "regulations")
    assert regulations.table == "regulation_chunks"
    assert regulations.rpc == "match_regulation_chunks"


def test_document_chunks_table():
    from retrieval.config import COLLECTIONS
    documents = next(c for c in COLLECTIONS if c.id == "documents")
    assert documents.table == "document_chunks"
    assert documents.rpc == "match_document_chunks"
