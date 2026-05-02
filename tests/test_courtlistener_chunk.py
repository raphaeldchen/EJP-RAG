"""
Test suite for chunk/courtlistener_chunk.py.

Unit tests  — synthetic pd.Series / dict inputs; no S3 access; test CL-specific logic
Corpus tests — full CL data from S3 via cl_opinion_chunks fixture (skip if unavailable)
S3 output   — reads actual opinion_chunks.jsonl from the chunked S3 bucket
"""

import dataclasses
from collections import defaultdict

import pandas as pd
import pytest

from chunk.courtlistener_chunk import (
    MAX_TOKENS,
    MIN_CHUNK_TOKENS,
    chunk_opinion,
    chunk_parentheticals,
    is_noise_chunk,
)
from core.models import Chunk

# ---------------------------------------------------------------------------
# Helpers — synthetic inputs
# ---------------------------------------------------------------------------


def _opinion_row(**kwargs) -> pd.Series:
    defaults = {
        "id": "42",
        "cluster_id": "10",
        "type": "020lead",
        "plain_text": "",
        "html_with_citations": "",
        "author_str": "Smith, J.",
        "per_curiam": "False",
    }
    return pd.Series({**defaults, **kwargs})


def _cluster(**kwargs) -> dict:
    defaults = {
        "id": "10",
        "docket_id": "5",
        "case_name": "United States v. Johnson",
        "case_name_short": "Johnson",
        "date_filed": "1995-03-15",
        "judges": "Smith, Brown",
        "precedential_status": "Published",
        "citation_count": "42",
    }
    return {**defaults, **kwargs}


def _docket(**kwargs) -> dict:
    defaults = {
        "id": "5",
        "court_id": "ca7",
        "docket_number": "94-1234",
        "nature_of_suit": "",
        "cause": "",
        "date_terminated": "",
    }
    return {**defaults, **kwargs}


_FILLER = "The court finds that the record fully supports the district court's conclusions on this matter. " * 20


# ---------------------------------------------------------------------------
# Unit tests — chunk_opinion
# ---------------------------------------------------------------------------


def test_empty_text_produces_no_chunks():
    row = _opinion_row(plain_text="", html_with_citations="")
    assert chunk_opinion(row, cluster_map={}, docket_map={}) == []


def test_short_opinion_produces_chunks():
    # Enough text to clear MIN_CHUNK_TOKENS (50) but small enough to fit in one chunk
    row = _opinion_row(
        plain_text=(
            "BACKGROUND\n\n"
            "Defendant was convicted of mail fraud under 18 U.S.C. § 1341. "
            "The district court sentenced him to 36 months imprisonment. "
            "On appeal, defendant argues the evidence was insufficient to support the conviction. "
            "The government responds that the evidence was overwhelming. "
            "We have reviewed the record and disagree with defendant. We affirm the judgment."
        )
    )
    chunks = chunk_opinion(row, {"10": _cluster()}, {"5": _docket()})
    assert chunks, "No chunks produced for non-empty opinion"


def test_returns_chunk_instances():
    row = _opinion_row(plain_text=_FILLER)
    chunks = chunk_opinion(row, cluster_map={}, docket_map={})
    for c in chunks:
        assert isinstance(c, Chunk), f"Expected Chunk, got {type(c)}"


def test_parent_id_equals_opinion_id():
    row = _opinion_row(id="opinion-99", plain_text=_FILLER)
    chunks = chunk_opinion(row, cluster_map={}, docket_map={})
    assert chunks
    assert all(c.parent_id == "opinion-99" for c in chunks)


def test_source_is_courtlistener():
    row = _opinion_row(plain_text=_FILLER)
    chunks = chunk_opinion(row, cluster_map={}, docket_map={})
    assert chunks
    assert all(c.source == "courtlistener" for c in chunks)


def test_chunk_indices_contiguous():
    row = _opinion_row(plain_text=_FILLER * 5)
    chunks = chunk_opinion(row, cluster_map={}, docket_map={})
    assert chunks
    indices = [c.chunk_index for c in chunks]
    assert sorted(indices) == list(range(len(indices)))
    assert all(c.chunk_total == len(chunks) for c in chunks)


def test_no_chunk_exceeds_max_tokens():
    row = _opinion_row(plain_text=_FILLER * 10)
    chunks = chunk_opinion(row, cluster_map={}, docket_map={})
    over = [(c.chunk_id, c.token_count) for c in chunks if c.token_count > MAX_TOKENS]
    assert not over, f"Chunks exceed MAX_TOKENS={MAX_TOKENS}: {over}"


def test_enriched_text_contains_case_name():
    cluster = _cluster(case_name_short="Johnson")
    row = _opinion_row(plain_text=_FILLER)
    chunks = chunk_opinion(row, {"10": cluster}, {"5": _docket()})
    assert chunks
    assert all("Johnson" in c.enriched_text for c in chunks)


def test_enriched_text_contains_chunk_text():
    row = _opinion_row(plain_text=_FILLER)
    chunks = chunk_opinion(row, cluster_map={}, docket_map={})
    assert chunks
    for c in chunks:
        assert c.text in c.enriched_text, "enriched_text must contain the raw chunk text"


def test_display_citation_contains_case_name_and_year():
    cluster = _cluster(case_name_short="Jones", date_filed="2001-06-15")
    row = _opinion_row(plain_text=_FILLER)
    chunks = chunk_opinion(row, {"10": cluster}, {"5": _docket()})
    assert chunks
    assert all("Jones" in c.display_citation for c in chunks)
    assert all("2001" in c.display_citation for c in chunks)


def test_metadata_contains_required_fields():
    row = _opinion_row(plain_text=_FILLER)
    chunks = chunk_opinion(row, {"10": _cluster()}, {"5": _docket()})
    assert chunks
    required_meta = {"opinion_id", "opinion_type", "case_name", "court_id", "chunk_type"}
    for c in chunks:
        missing = required_meta - c.metadata.keys()
        assert not missing, f"Chunk {c.chunk_id} metadata missing: {missing}"


def test_html_fallback_when_no_plain_text():
    body = "We affirm the conviction of the defendant on all counts of the indictment. " * 8
    html = f"<p>BACKGROUND</p><p>{body}</p>"
    row = _opinion_row(plain_text="", html_with_citations=html)
    chunks = chunk_opinion(row, cluster_map={}, docket_map={})
    assert chunks, "Expected chunks from HTML fallback"


def test_chunk_ids_unique():
    row = _opinion_row(id="op-1", plain_text=_FILLER * 5)
    chunks = chunk_opinion(row, cluster_map={}, docket_map={})
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk IDs"


def test_chunk_id_contains_opinion_id():
    row = _opinion_row(id="op-77", plain_text=_FILLER)
    chunks = chunk_opinion(row, cluster_map={}, docket_map={})
    assert chunks
    assert all("op-77" in c.chunk_id for c in chunks)


def test_asdict_round_trips():
    """Chunk.asdict() must produce a JSON-serialisable dict (no pd.Series, etc.)."""
    row = _opinion_row(plain_text=_FILLER)
    chunks = chunk_opinion(row, {"10": _cluster()}, {"5": _docket()})
    assert chunks
    import json
    d = dataclasses.asdict(chunks[0])
    json.dumps(d)  # raises if any value is not JSON-serialisable


def test_metadata_opinion_id_matches_parent_id():
    row = _opinion_row(id="op-X", plain_text=_FILLER)
    chunks = chunk_opinion(row, cluster_map={}, docket_map={})
    assert chunks
    for c in chunks:
        assert c.metadata.get("opinion_id") == c.parent_id


def test_no_min_chunk_token_violations():
    """Noise filter must discard very short fragments."""
    row = _opinion_row(plain_text=_FILLER)
    chunks = chunk_opinion(row, cluster_map={}, docket_map={})
    assert chunks
    assert all(c.token_count >= MIN_CHUNK_TOKENS for c in chunks)


# ---------------------------------------------------------------------------
# Unit tests — chunk_parentheticals
# ---------------------------------------------------------------------------


def test_parenthetical_empty_df_produces_no_chunks():
    df = pd.DataFrame(
        columns=["id", "text", "describing_opinion_id", "described_opinion_id", "score"]
    )
    result = chunk_parentheticals(df, cluster_map={}, docket_map={}, opinion_to_cluster={})
    assert result == []


def test_parenthetical_chunk_fields():
    df = pd.DataFrame([{
        "id": "par-1",
        "text": "holding that malice aforethought requires intent to kill",
        "describing_opinion_id": "op-42",
        "described_opinion_id": "op-7",
        "score": "0.82",
    }])
    cluster = _cluster(id="cl-5", docket_id="dk-3", case_name="State v. Smith", date_filed="2010-01-01")
    docket = _docket(id="dk-3")
    chunks = chunk_parentheticals(df, {"cl-5": cluster}, {"dk-3": docket}, {"op-42": "cl-5"})
    assert len(chunks) == 1
    c = chunks[0]
    assert isinstance(c, Chunk)
    assert c.chunk_id == "par_par-1"
    assert c.parent_id == "op-42"
    assert c.chunk_index == 0
    assert c.chunk_total == 1
    assert c.source == "courtlistener"
    assert "malice aforethought" in c.text
    assert c.metadata.get("chunk_type") == "parenthetical"


def test_parenthetical_enriched_text_contains_case_name():
    df = pd.DataFrame([{
        "id": "par-2",
        "text": "holding that the search was unreasonable under the Fourth Amendment",
        "describing_opinion_id": "op-10",
        "described_opinion_id": "op-5",
        "score": "0.9",
    }])
    cluster = _cluster(id="cl-1", docket_id="dk-1", case_name="Brown v. Board", date_filed="2005-04-01")
    docket = _docket(id="dk-1")
    chunks = chunk_parentheticals(df, {"cl-1": cluster}, {"dk-1": docket}, {"op-10": "cl-1"})
    assert chunks
    assert "Brown" in chunks[0].enriched_text


def test_parenthetical_display_citation():
    df = pd.DataFrame([{
        "id": "par-3",
        "text": "holding that the right to counsel attaches at arraignment",
        "describing_opinion_id": "op-5",
        "described_opinion_id": "op-2",
        "score": "0.75",
    }])
    cluster = _cluster(id="cl-2", docket_id="dk-2", case_name="People v. Davis", date_filed="2012-11-20")
    docket = _docket(id="dk-2")
    chunks = chunk_parentheticals(df, {"cl-2": cluster}, {"dk-2": docket}, {"op-5": "cl-2"})
    assert chunks
    assert "Davis" in chunks[0].display_citation or "People" in chunks[0].display_citation


def test_parenthetical_skips_empty_text():
    df = pd.DataFrame([
        {"id": "par-1", "text": "  ", "describing_opinion_id": "op-1",
         "described_opinion_id": "op-2", "score": "0.5"},
        {"id": "par-2", "text": "valid substantive holding text here",
         "describing_opinion_id": "op-3", "described_opinion_id": "op-4", "score": "0.6"},
    ])
    chunks = chunk_parentheticals(df, {}, {}, {})
    assert len(chunks) == 1
    assert "valid substantive" in chunks[0].text


def test_parenthetical_metadata_contains_describing_and_described_ids():
    df = pd.DataFrame([{
        "id": "par-4",
        "text": "holding that the statute is unconstitutional as applied",
        "describing_opinion_id": "op-A",
        "described_opinion_id": "op-B",
        "score": "0.88",
    }])
    chunks = chunk_parentheticals(df, {}, {}, {})
    assert chunks
    meta = chunks[0].metadata
    assert meta.get("describing_opinion_id") == "op-A"
    assert meta.get("described_opinion_id") == "op-B"


# ---------------------------------------------------------------------------
# Corpus-level property tests (require S3 via cl_opinion_chunks fixture)
# ---------------------------------------------------------------------------


def test_corpus_no_chunk_exceeds_max_tokens(cl_opinion_chunks):
    failures = [
        (c["chunk_id"], c["token_count"])
        for c in cl_opinion_chunks
        if c["token_count"] > MAX_TOKENS
    ]
    assert not failures, f"{len(failures)} chunks exceed MAX_TOKENS: {failures[:5]}"


def test_corpus_chunk_index_contiguous(cl_opinion_chunks):
    by_parent = defaultdict(list)
    for c in cl_opinion_chunks:
        by_parent[c["parent_id"]].append(c["chunk_index"])
    failures = [
        f"{pid}: {sorted(idxs)}"
        for pid, idxs in by_parent.items()
        if set(idxs) != set(range(len(idxs)))
    ]
    assert not failures, "Non-contiguous chunk_index values:\n" + "\n".join(failures[:5])


def test_corpus_chunk_total_accurate(cl_opinion_chunks):
    by_parent = defaultdict(list)
    for c in cl_opinion_chunks:
        by_parent[c["parent_id"]].append(c)
    failures = [
        f"{c['chunk_id']}: chunk_total={c['chunk_total']} actual={len(siblings)}"
        for siblings in by_parent.values()
        for c in siblings
        if c["chunk_total"] != len(siblings)
    ]
    assert not failures, "chunk_total mismatches:\n" + "\n".join(failures[:5])


def test_corpus_no_empty_chunks(cl_opinion_chunks):
    failures = [c["chunk_id"] for c in cl_opinion_chunks if c["token_count"] < MIN_CHUNK_TOKENS]
    assert not failures, f"{len(failures)} chunks below MIN_CHUNK_TOKENS: {failures[:5]}"


def test_corpus_chunk_ids_unique(cl_opinion_chunks):
    ids = [c["chunk_id"] for c in cl_opinion_chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk_ids"


def test_corpus_all_chunks_have_required_fields(cl_opinion_chunks):
    required = {"chunk_id", "parent_id", "chunk_index", "chunk_total", "source",
                "text", "enriched_text", "token_count"}
    failures = [
        f"{c['chunk_id']}: missing {required - c.keys()}"
        for c in cl_opinion_chunks
        if not required.issubset(c.keys())
    ]
    assert not failures, "Chunks missing required fields:\n" + "\n".join(failures[:5])


def test_corpus_source_field(cl_opinion_chunks):
    failures = [c["chunk_id"] for c in cl_opinion_chunks if c.get("source") != "courtlistener"]
    assert not failures, f"{len(failures)} chunks with unexpected source: {failures[:5]}"


# ---------------------------------------------------------------------------
# S3 output verification
# ---------------------------------------------------------------------------


def test_s3_output_no_corrupt_records(cl_opinion_chunks_s3):
    required = {"chunk_id", "parent_id", "chunk_index", "chunk_total", "source",
                "text", "enriched_text", "token_count"}
    failures = [
        f"record {i}: missing {required - c.keys()}"
        for i, c in enumerate(cl_opinion_chunks_s3)
        if not required.issubset(c.keys())
    ]
    assert not failures, "Corrupt records in S3 output:\n" + "\n".join(failures[:5])


def test_s3_output_no_empty_text(cl_opinion_chunks_s3):
    failures = [c["chunk_id"] for c in cl_opinion_chunks_s3 if not c.get("text", "").strip()]
    assert not failures, f"{len(failures)} chunks with empty text in S3: {failures[:5]}"


def test_s3_output_source_field(cl_opinion_chunks_s3):
    failures = [c["chunk_id"] for c in cl_opinion_chunks_s3 if c.get("source") != "courtlistener"]
    assert not failures, f"{len(failures)} S3 chunks with unexpected source: {failures[:5]}"
