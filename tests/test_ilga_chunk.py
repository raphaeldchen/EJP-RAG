import re
import pytest
from chunk.ilga_chunk import chunk_section, CHUNK_SIZE, MIN_CHUNK_SIZE


def test_5_915_chunk_is_self_contained(ilcs_records):
    """705 ILCS 405/5-915 must produce at least one chunk with key expungement terms."""
    record = next(
        (r for r in ilcs_records if "405/5-915" in r.get("section_citation", "")),
        None,
    )
    if record is None:
        pytest.skip("705 ILCS 405/5-915 not in corpus — run ilga_ingest.py --chapters 705 first")
    chunks = chunk_section(record)
    assert chunks, "No chunks produced for 5-915"
    assert chunks[0].display_citation  # non-empty
    assert " — " in chunks[0].display_citation  # citation — heading format
    assert chunks[0].token_count > 0
    assert chunks[0].source == "ilcs"
    assert "section_citation" in chunks[0].metadata
    assert chunks[0].enriched_text  # non-empty (ILCS now produces enriched_text)
    texts = [c.text for c in chunks]
    has_term = any(re.search(r"expunge|automatic", t, re.IGNORECASE) for t in texts)
    assert has_term, "No chunk contains 'expunge' or 'automatic' — section may be split incorrectly"
    for chunk in chunks:
        last_char = chunk.text.rstrip()[-1]
        assert last_char in ".!?;:\"')", (
            f"Chunk {chunk.chunk_id} ends mid-word (last char: {last_char!r})"
        )


def test_enumeration_not_severed():
    """A section long enough to trigger splitting must not orphan subsection markers."""
    filler = "Word " * 80  # ~400 chars per subsection
    text = (
        "The court shall consider all of the following factors:\n"
        f"(a) {filler}\n"
        f"(b) {filler}\n"
        f"(c) {filler}\n"
    )
    record = {
        "id": "test_enum",
        "text": text,
        "section_citation": "TEST/1-1",
        "section_heading": "Test Section",
        "section_num": "1-1",
        "article_name": "",
        "act_name": "",
        "act_id": "",
        "chapter_num": "",
        "chapter_name": "",
        "major_topic": "",
        "url": "",
        "scraped_at": "",
    }
    chunks = chunk_section(record)
    orphan_re = re.compile(r"^\s*\([a-z0-9]\)")
    for chunk in chunks:
        if chunk.chunk_index > 0:
            assert not orphan_re.match(chunk.text), (
                f"Chunk {chunk.chunk_id} (index {chunk.chunk_index}) "
                f"starts with orphaned subsection marker:\n{chunk.text[:100]}"
            )


from collections import defaultdict


def test_no_orphaned_subsection_starts(ilcs_chunks):
    orphan_re = re.compile(r"^\s*\([a-z0-9]\)")
    failures = []
    for chunk in ilcs_chunks:
        if chunk.chunk_index > 0 and orphan_re.match(chunk.text):
            failures.append(chunk.chunk_id)
    assert not failures, (
        f"{len(failures)} chunks start with orphaned subsection markers: {failures[:5]}"
    )


def test_chunk_index_contiguous(ilcs_chunks):
    by_parent = defaultdict(list)
    for chunk in ilcs_chunks:
        by_parent[chunk.parent_id].append(chunk.chunk_index)
    failures = []
    for parent_id, indices in by_parent.items():
        expected = set(range(len(indices)))
        if set(indices) != expected:
            failures.append(f"{parent_id}: {sorted(indices)}")
    assert not failures, f"Non-contiguous chunk indices:\n" + "\n".join(failures[:5])


def test_chunk_total_accurate(ilcs_chunks):
    by_parent = defaultdict(list)
    for chunk in ilcs_chunks:
        by_parent[chunk.parent_id].append(chunk)
    failures = []
    for parent_id, siblings in by_parent.items():
        actual = len(siblings)
        for chunk in siblings:
            if chunk.chunk_total != actual:
                failures.append(
                    f"{chunk.chunk_id}: chunk_total={chunk.chunk_total} actual={actual}"
                )
    assert not failures, f"chunk_total mismatches:\n" + "\n".join(failures[:5])


def test_no_empty_chunks(ilcs_chunks):
    failures = [c.chunk_id for c in ilcs_chunks if len(c.text) < MIN_CHUNK_SIZE]
    assert not failures, f"{len(failures)} chunks below MIN_CHUNK_SIZE: {failures[:5]}"


def test_chunk_ids_unique(ilcs_chunks):
    ids = [c.chunk_id for c in ilcs_chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk_ids found in ILCS chunks"


def test_no_chunk_exceeds_max_size(ilcs_chunks):
    failures = [
        (c.chunk_id, len(c.text))
        for c in ilcs_chunks
        if len(c.text) > CHUNK_SIZE
    ]
    assert not failures, (
        f"{len(failures)} chunks exceed CHUNK_SIZE={CHUNK_SIZE}: {failures[:5]}"
    )


def test_sentence_split_overlap(ilcs_chunks):  # noqa: F811
    """For sentence-split parents, chunk N+1 should contain the last sentence of chunk N."""
    from chunk.ilga_chunk import CHUNK_OVERLAP
    by_parent = defaultdict(list)
    for chunk in ilcs_chunks:
        by_parent[chunk.parent_id].append(chunk)

    sentence_end_re = re.compile(r"(?<=[.!?])\s+")
    subsection_re = re.compile(r"^\([a-z0-9]\)", re.MULTILINE)
    failures = []

    for parent_id, siblings in by_parent.items():
        if len(siblings) <= 1:
            continue
        siblings_sorted = sorted(siblings, key=lambda c: c.chunk_index)
        # Only check sentence-split parents (no subsection markers in any chunk)
        if any(subsection_re.search(c.text) for c in siblings_sorted):
            continue
        for i in range(len(siblings_sorted) - 1):
            chunk_a = siblings_sorted[i]
            chunk_b = siblings_sorted[i + 1]
            sentences = sentence_end_re.split(chunk_a.text)
            last_sentence = sentences[-1].strip() if sentences else ""
            if len(last_sentence) < 15:
                continue
            # When the "last sentence" is a giant fragment (no terminal punctuation), fall back
            # to checking that a tail suffix of chunk A appears in chunk B — hard-split overlap.
            if len(last_sentence) > CHUNK_OVERLAP:
                tail = chunk_a.text[-CHUNK_OVERLAP:].strip()
                if tail and tail not in chunk_b.text:
                    failures.append(
                        f"No tail overlap: {chunk_a.chunk_id} → {chunk_b.chunk_id}"
                    )
                continue
            if last_sentence not in chunk_b.text:
                failures.append(
                    f"No overlap: {chunk_a.chunk_id} → {chunk_b.chunk_id}"
                )

    assert not failures, (
        f"{len(failures)} consecutive chunk pairs have no overlap:\n"
        + "\n".join(failures[:5])
    )


def test_chunk_schema_fields(ilcs_chunks):
    """All chunks must have the required Chunk dataclass fields populated."""
    failures = []
    for c in ilcs_chunks:
        if not c.display_citation:
            failures.append(f"{c.chunk_id}: display_citation empty")
        if " — " not in c.display_citation and c.metadata.get("section_heading"):
            failures.append(f"{c.chunk_id}: display_citation missing ' — ' separator")
        if c.token_count <= 0:
            failures.append(f"{c.chunk_id}: token_count={c.token_count}")
        if c.source != "ilcs":
            failures.append(f"{c.chunk_id}: source={c.source!r}")
        if "section_citation" not in c.metadata:
            failures.append(f"{c.chunk_id}: metadata missing section_citation")
        if not c.enriched_text:
            failures.append(f"{c.chunk_id}: enriched_text empty")
        if not c.enriched_text.endswith(c.text):
            failures.append(f"{c.chunk_id}: enriched_text does not end with chunk text")
        if c.text not in c.enriched_text:
            failures.append(f"{c.chunk_id}: chunk text not present in enriched_text")
    assert not failures, f"{len(failures)} schema violations:\n" + "\n".join(failures[:5])


# ---------------------------------------------------------------------------
# S3 output verification — reads the actual chunked file from the chunked bucket
# ---------------------------------------------------------------------------

def test_s3_output_record_count(ilcs_chunks, ilcs_chunks_s3):
    """Chunk count in S3 must match what the chunker produces in memory."""
    assert len(ilcs_chunks_s3) == len(ilcs_chunks), (
        f"S3 has {len(ilcs_chunks_s3)} chunks, in-memory produced {len(ilcs_chunks)}"
    )


def test_s3_output_no_corrupt_records(ilcs_chunks_s3):
    """Every record in S3 must have the required top-level keys."""
    required = {"chunk_id", "chunk_index", "chunk_total", "text", "metadata"}
    failures = [
        f"record {i}: missing {required - c.keys()}"
        for i, c in enumerate(ilcs_chunks_s3)
        if not required.issubset(c.keys())
    ]
    assert not failures, "Corrupt records in S3 output:\n" + "\n".join(failures[:5])


def test_s3_output_no_empty_text(ilcs_chunks_s3):
    """No chunk in S3 should have empty text."""
    failures = [c["chunk_id"] for c in ilcs_chunks_s3 if not c.get("text", "").strip()]
    assert not failures, f"{len(failures)} chunks with empty text in S3: {failures[:5]}"


def test_s3_output_source_is_ilcs(ilcs_chunks_s3):
    """All S3 chunks must have source=ilcs in metadata."""
    failures = [
        c["chunk_id"] for c in ilcs_chunks_s3
        if c.get("metadata", {}).get("source") != "ilcs"
    ]
    assert not failures, f"{len(failures)} chunks with wrong source in S3: {failures[:5]}"
