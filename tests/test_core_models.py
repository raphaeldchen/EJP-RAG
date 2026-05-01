import dataclasses
from core.models import Chunk


def test_chunk_required_fields():
    c = Chunk(
        chunk_id="test_c0",
        parent_id="test",
        chunk_index=0,
        chunk_total=1,
        text="Some legal text.",
        enriched_text="720 ILCS 5/7-1 — Justifiable Use of Force\n\nSome legal text.",
        source="ilcs",
        token_count=3,
        display_citation="720 ILCS 5/7-1 — Justifiable Use of Force",
    )
    assert c.chunk_id == "test_c0"
    assert c.metadata == {}
    assert c.chunked_at  # auto-populated


def test_chunk_asdict_round_trips():
    c = Chunk(
        chunk_id="test_c0", parent_id="test", chunk_index=0, chunk_total=1,
        text="text", enriched_text="enriched", source="ilcs",
        token_count=1, display_citation="Citation",
    )
    d = dataclasses.asdict(c)
    assert d["chunk_id"] == "test_c0"
    assert d["display_citation"] == "Citation"
    assert "metadata" in d
