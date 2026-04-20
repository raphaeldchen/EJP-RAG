import re
import pytest
from chunk.ilga_chunk import chunk_section, CHUNK_SIZE, MIN_CHUNK_SIZE


def test_5_915_chunk_is_self_contained(ilcs_records):
    """705 ILCS 405/5-915 must produce at least one chunk with key expungement terms."""
    record = next(
        (r for r in ilcs_records if "405/5-915" in r.get("section_citation", "")),
        None,
    )
    assert record is not None, "705 ILCS 405/5-915 record not found in corpus"
    chunks = chunk_section(record)
    assert chunks, "No chunks produced for 5-915"
    texts = [c["text"] for c in chunks]
    has_term = any(re.search(r"expunge|automatic", t, re.IGNORECASE) for t in texts)
    assert has_term, "No chunk contains 'expunge' or 'automatic' — section may be split incorrectly"
    for chunk in chunks:
        last_char = chunk["text"].rstrip()[-1]
        assert last_char in ".!?;:\"')", (
            f"Chunk {chunk['chunk_id']} ends mid-word (last char: {last_char!r})"
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
        if chunk["chunk_index"] > 0:
            assert not orphan_re.match(chunk["text"]), (
                f"Chunk {chunk['chunk_id']} (index {chunk['chunk_index']}) "
                f"starts with orphaned subsection marker:\n{chunk['text'][:100]}"
            )
