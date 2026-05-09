"""
Tests for the CAP bulk opinion chunker (chunk/cap_chunk.py).

S3 output verification only — the full corpus is ~6.4 GB / 1.2M records,
so the cap_chunks_s3 fixture samples ~10,000–20,000 records from five
evenly-spaced positions across the file rather than loading it entirely.
The core split/detect logic is shared with courtlistener_chunk and is
covered by test_courtlistener_chunk.py.
"""

import os
import re

import boto3
import pytest
from dotenv import load_dotenv

load_dotenv()

_VALID_OPINION_TYPES = {
    "majority",
    "dissent",
    "concurrence",
    "rehearing",
    "preamble",
    "addendum",
    "remittitur",
}

_CHUNK_ID_RE = re.compile(r"^.+_(majority|dissent|concurrence|rehearing|preamble|addendum|remittitur)\d+_c\d+$")

_REQUIRED_FIELDS = {
    "chunk_id", "parent_id", "chunk_index", "chunk_total",
    "source", "text", "enriched_text", "token_count", "display_citation", "metadata",
}

_REQUIRED_METADATA = {
    "opinion_type", "is_majority", "case_id", "case_name",
    "date_decided", "court", "court_label", "citations",
}


# ---------------------------------------------------------------------------
# S3 file-level sanity check (no fixture needed)
# ---------------------------------------------------------------------------

def test_s3_file_exists_and_complete():
    """CAP chunked file must exist in S3 and be at least 5 GB (proxy for a full write)."""
    bucket = os.environ.get("CHUNKED_S3_BUCKET")
    if not bucket:
        pytest.skip("CHUNKED_S3_BUCKET not set")
    cap_prefix = os.environ.get("CAP_S3_PREFIX", "cap").rstrip("/")
    key = f"{cap_prefix}/cap_opinion_chunks.jsonl"
    s3 = boto3.client("s3")
    try:
        meta = s3.head_object(Bucket=bucket, Key=key)
    except Exception as e:
        pytest.fail(f"CAP chunks file not found in S3: {e}")
    size_gb = meta["ContentLength"] / (1024 ** 3)
    assert size_gb >= 5.0, f"CAP chunks file is only {size_gb:.2f} GB — may be truncated"


# ---------------------------------------------------------------------------
# S3 output verification (sampled)
# ---------------------------------------------------------------------------

def test_s3_no_corrupt_records(cap_chunks_s3):
    """Every sampled record must have all required top-level fields."""
    failures = [
        f"record {i} ({c.get('chunk_id', '?')}): missing {_REQUIRED_FIELDS - c.keys()}"
        for i, c in enumerate(cap_chunks_s3)
        if not _REQUIRED_FIELDS.issubset(c.keys())
    ]
    assert not failures, "Corrupt records in S3 sample:\n" + "\n".join(failures[:5])


def test_s3_no_corrupt_metadata(cap_chunks_s3):
    """Every sampled record's metadata dict must have all required metadata fields."""
    failures = [
        f"{c['chunk_id']}: missing metadata keys {_REQUIRED_METADATA - c['metadata'].keys()}"
        for c in cap_chunks_s3
        if not _REQUIRED_METADATA.issubset(c.get("metadata", {}).keys())
    ]
    assert not failures, "Records with missing metadata keys:\n" + "\n".join(failures[:5])


def test_s3_no_empty_text(cap_chunks_s3):
    failures = [c["chunk_id"] for c in cap_chunks_s3 if not c.get("text", "").strip()]
    assert not failures, f"{len(failures)} chunks with empty text: {failures[:5]}"


def test_s3_no_empty_enriched_text(cap_chunks_s3):
    failures = [c["chunk_id"] for c in cap_chunks_s3 if not c.get("enriched_text", "").strip()]
    assert not failures, f"{len(failures)} chunks with empty enriched_text: {failures[:5]}"


def test_s3_source_is_cap_bulk(cap_chunks_s3):
    failures = [c["chunk_id"] for c in cap_chunks_s3 if c.get("source") != "cap_bulk"]
    assert not failures, f"{len(failures)} chunks with wrong source: {failures[:5]}"


def test_s3_opinion_types_valid(cap_chunks_s3):
    """metadata.opinion_type must be one of the normalized CAP opinion types."""
    failures = [
        f"{c['chunk_id']}: {c['metadata'].get('opinion_type')!r}"
        for c in cap_chunks_s3
        if c.get("metadata", {}).get("opinion_type") not in _VALID_OPINION_TYPES
    ]
    assert not failures, f"{len(failures)} chunks with invalid opinion_type:\n" + "\n".join(failures[:5])


def test_s3_is_majority_is_bool(cap_chunks_s3):
    """metadata.is_majority must be a bool, not a string or None."""
    failures = [
        f"{c['chunk_id']}: {c['metadata'].get('is_majority')!r} ({type(c['metadata'].get('is_majority')).__name__})"
        for c in cap_chunks_s3
        if not isinstance(c.get("metadata", {}).get("is_majority"), bool)
    ]
    assert not failures, f"{len(failures)} chunks with non-bool is_majority:\n" + "\n".join(failures[:5])


def test_s3_chunk_id_format(cap_chunks_s3):
    """chunk_id must end with _{opinion_type}_c{n}."""
    failures = [
        c["chunk_id"]
        for c in cap_chunks_s3
        if not _CHUNK_ID_RE.match(c.get("chunk_id", ""))
    ]
    assert not failures, f"{len(failures)} chunks with malformed chunk_id:\n" + "\n".join(failures[:5])


def test_s3_chunk_index_non_negative(cap_chunks_s3):
    failures = [c["chunk_id"] for c in cap_chunks_s3 if c.get("chunk_index", -1) < 0]
    assert not failures, f"{len(failures)} chunks with negative chunk_index: {failures[:5]}"


def test_s3_token_count_positive(cap_chunks_s3):
    failures = [c["chunk_id"] for c in cap_chunks_s3 if c.get("token_count", 0) <= 0]
    assert not failures, f"{len(failures)} chunks with zero/negative token_count: {failures[:5]}"


def test_s3_preamble_not_marked_majority(cap_chunks_s3):
    """Preamble chunks must have is_majority=False (preamble type is not in _MAJORITY_TYPES)."""
    failures = [
        c["chunk_id"]
        for c in cap_chunks_s3
        if c.get("metadata", {}).get("opinion_type") == "preamble"
        and c["metadata"].get("is_majority") is True
    ]
    assert not failures, f"{len(failures)} preamble chunks incorrectly flagged is_majority=True: {failures[:5]}"


def test_s3_majority_chunks_flagged_correctly(cap_chunks_s3):
    """Chunks with opinion_type='majority' must have is_majority=True."""
    failures = [
        c["chunk_id"]
        for c in cap_chunks_s3
        if c.get("metadata", {}).get("opinion_type") == "majority"
        and c["metadata"].get("is_majority") is not True
    ]
    assert not failures, f"{len(failures)} majority chunks with is_majority != True: {failures[:5]}"


def test_s3_chunk_ids_unique(cap_chunks_s3):
    """All chunk_ids in the sampled S3 output must be unique."""
    ids = [c["chunk_id"] for c in cap_chunks_s3]
    duplicates = [cid for cid, count in __import__("collections").Counter(ids).items() if count > 1]
    assert not duplicates, f"{len(duplicates)} duplicate chunk_ids: {duplicates[:5]}"


# ---------------------------------------------------------------------------
# In-memory unit tests (no S3 needed)
# ---------------------------------------------------------------------------

def test_chunk_entry_unique_ids_with_two_concurrences():
    """Two [Concurrence] / [Concur] segments in the same case must not collide."""
    from chunk.cap_chunk import chunk_entry

    entry = {
        "id": "cap_test_001",
        "case_id": "001",
        "case_name": "People v. Test",
        "case_name_abbr": "People v. Test",
        "date_decided": "2000-01-01",
        "court": "ill-2d",
        "citations": ["100 Ill. 2d 1"],
        "text": (
            "[Majority] The court holds that the statute is constitutional. "
            "Defendant's argument fails because the plain text is unambiguous. "
            "Affirmed.\n"
            "[Concurrence] Justice A concurs in the judgment but writes separately "
            "to address the due process argument raised by the dissent.\n"
            "[Concur] Justice B concurs only in part and would remand for resentencing."
        ),
    }
    chunks = chunk_entry(entry)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), f"Duplicate chunk_ids: {[x for x in ids if ids.count(x) > 1]}"


def test_chunk_entry_unique_ids_with_two_dissents():
    """Two [Dissent] segments (different justices) must not collide."""
    from chunk.cap_chunk import chunk_entry

    entry = {
        "id": "cap_test_002",
        "case_id": "002",
        "case_name": "People v. Smith",
        "case_name_abbr": "People v. Smith",
        "date_decided": "2005-06-15",
        "court": "ill-2d",
        "citations": ["200 Ill. 2d 50"],
        "text": (
            "[Majority] The conviction is affirmed. The evidence was sufficient "
            "to support the jury's verdict beyond a reasonable doubt.\n"
            "[Dissent] Justice X dissents and would reverse the conviction outright "
            "because the search was unconstitutional.\n"
            "[Dissent] Justice Y also dissents but on narrower grounds, arguing only "
            "that the sentence was excessive under the proportionality clause."
        ),
    }
    chunks = chunk_entry(entry)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), f"Duplicate chunk_ids: {[x for x in ids if ids.count(x) > 1]}"
