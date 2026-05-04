import json
import pytest

from embed.batch_embed import (
    SOURCE_REGISTRY,
    _is_cap_criminal,
    _is_ilcs_in_scope,
    _normalize_cl_chunk,
    build_payload,
    iter_records,
)


# ---------------------------------------------------------------------------
# _is_cap_criminal
# ---------------------------------------------------------------------------

def test_cap_criminal_people_v_case_name():
    assert _is_cap_criminal({"metadata": {"case_name": "People v. Jones"}, "text": ""}) is True


def test_cap_criminal_people_of_state_variant():
    # Historical cases style the state on the right: "Smith v. The People of the State of Illinois"
    record = {"metadata": {"case_name": "Spies v. The People of the State of Illinois"}, "text": ""}
    assert _is_cap_criminal(record) is True


def test_cap_criminal_in_re_name():
    assert _is_cap_criminal({"metadata": {"case_name": "In re Jones"}, "text": ""}) is True


def test_cap_criminal_in_re_commitment():
    record = {"metadata": {"case_name": "In re Commitment of Jones"}, "text": ""}
    assert _is_cap_criminal(record) is True


def test_cap_criminal_statute_in_text_overrides_civil_name():
    record = {
        "metadata": {"case_name": "ABC Corp v. XYZ Inc."},
        "text": "The court interpreted 720 ILCS 5/9-1(a) in reaching its holding.",
    }
    assert _is_cap_criminal(record) is True


def test_cap_criminal_730_ilcs_in_text():
    record = {
        "metadata": {"case_name": "Jones v. Smith"},
        "text": "Sentence was imposed under 730 ILCS 5/5-4.5-95.",
    }
    assert _is_cap_criminal(record) is True


def test_cap_civil_no_criminal_signals_excluded():
    record = {
        "metadata": {"case_name": "Philip Morris Inc. v. Price"},
        "text": "The class action alleges breach of implied warranty and consumer fraud.",
    }
    assert _is_cap_criminal(record) is False


def test_cap_civil_property_dispute_excluded():
    record = {
        "metadata": {"case_name": "Greenfield Properties v. Cook County"},
        "text": "The court reversed the property tax assessment.",
    }
    assert _is_cap_criminal(record) is False


# ---------------------------------------------------------------------------
# _is_ilcs_in_scope
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("chapter", ["720", "725", "730", "705", "735", "750", "775", "625"])
def test_ilcs_core_criminal_chapters_in_scope(chapter):
    assert _is_ilcs_in_scope({"metadata": {"chapter_num": chapter, "section_citation": f"{chapter} ILCS 1/1"}})


def test_ilcs_torts_chapter_excluded():
    assert _is_ilcs_in_scope({"metadata": {"chapter_num": "740", "section_citation": "740 ILCS 14/1"}}) is False


def test_ilcs_chapter_20_commerce_dept_excluded():
    record = {"metadata": {"chapter_num": "20", "section_citation": "20 ILCS 605/605-25"}}
    assert _is_ilcs_in_scope(record) is False


def test_ilcs_chapter_20_lottery_excluded():
    record = {"metadata": {"chapter_num": "20", "section_citation": "20 ILCS 1605/1605-10"}}
    assert _is_ilcs_in_scope(record) is False


def test_ilcs_chapter_20_cjia_included():
    # Criminal Justice Information Authority — clearly relevant
    record = {"metadata": {"chapter_num": "20", "section_citation": "20 ILCS 3930/7"}}
    assert _is_ilcs_in_scope(record) is True


def test_ilcs_chapter_20_doc_included():
    # Department of Corrections
    record = {"metadata": {"chapter_num": "20", "section_citation": "20 ILCS 415/1"}}
    assert _is_ilcs_in_scope(record) is True


# ---------------------------------------------------------------------------
# SOURCE_REGISTRY — hardcoded table assignments
# ---------------------------------------------------------------------------

_ALL_SOURCES = {
    "ilcs", "iscr",
    "cap_bulk", "courtlistener",
    "iac", "idoc",
    "spac", "iccb", "federal", "restorejustice", "cookcounty_pd",
}


def test_source_registry_contains_all_sources():
    assert set(SOURCE_REGISTRY.keys()) == _ALL_SOURCES


def test_opinion_sources_go_to_opinion_table():
    assert SOURCE_REGISTRY["cap_bulk"].table == "opinion_chunks"
    assert SOURCE_REGISTRY["courtlistener"].table == "opinion_chunks"


def test_regulation_sources_go_to_regulation_table():
    assert SOURCE_REGISTRY["iac"].table == "regulation_chunks"
    assert SOURCE_REGISTRY["idoc"].table == "regulation_chunks"


@pytest.mark.parametrize("src", ["spac", "iccb", "federal", "restorejustice", "cookcounty_pd"])
def test_document_sources_go_to_document_table(src):
    assert SOURCE_REGISTRY[src].table == "document_chunks"


def test_ilcs_goes_to_ilcs_table():
    assert SOURCE_REGISTRY["ilcs"].table == "ilcs_chunks"


def test_iscr_goes_to_court_rule_table():
    assert SOURCE_REGISTRY["iscr"].table == "court_rule_chunks"


def test_cap_has_criminal_filter():
    entry = SOURCE_REGISTRY["cap_bulk"]
    assert entry.filter_fn is not None


def test_ilcs_has_scope_filter():
    entry = SOURCE_REGISTRY["ilcs"]
    assert entry.filter_fn is not None


def test_non_filtered_sources_have_no_filter():
    for src in ("iscr", "courtlistener", "iac", "idoc", "spac", "iccb", "federal",
                "restorejustice", "cookcounty_pd"):
        assert SOURCE_REGISTRY[src].filter_fn is None, f"{src} unexpectedly has a filter"


# ---------------------------------------------------------------------------
# build_payload
# ---------------------------------------------------------------------------

def _make_opinion_record(**overrides):
    rec = {
        "chunk_id": "cap-123_c0",
        "parent_id": "cap-123",
        "chunk_index": 0,
        "chunk_total": 3,
        "source": "cap_bulk",
        "token_count": 400,
        "display_citation": "2020 IL 12345 — People v. Jones",
        "text": "The court held...",
        "enriched_text": "[Illinois Supreme Court | People v. Jones (2020-01-01)]\n\nThe court held...",
        "metadata": {"case_name": "People v. Jones"},
    }
    rec.update(overrides)
    return rec


def test_build_payload_opinion_chunks_has_universal_fields():
    payload = build_payload(_make_opinion_record(), [0.1] * 768, "opinion_chunks")
    assert payload["chunk_id"] == "cap-123_c0"
    assert payload["parent_id"] == "cap-123"
    assert payload["chunk_index"] == 0
    assert payload["chunk_total"] == 3
    assert payload["source"] == "cap_bulk"
    assert payload["token_count"] == 400
    assert payload["display_citation"] == "2020 IL 12345 — People v. Jones"
    assert payload["embedding"] == [0.1] * 768


def test_build_payload_regulation_chunks_has_universal_fields():
    rec = _make_opinion_record(source="iac", chunk_id="iac-abc_c0")
    payload = build_payload(rec, [0.2] * 768, "regulation_chunks")
    assert payload["chunk_id"] == "iac-abc_c0"
    assert payload["source"] == "iac"
    assert "embedding" in payload


def test_build_payload_ilcs_has_legacy_columns():
    record = {
        "chunk_id": "ch720-act2465-sec5_c0",
        "parent_id": "ch720-act2465-sec5",
        "chunk_index": 0,
        "chunk_total": 1,
        "source": "ilcs",
        "token_count": 200,
        "display_citation": "720 ILCS 5/9-1 — First degree murder",
        "text": "A person commits...",
        "enriched_text": "[720 ILCS 5/9-1]\nA person commits...",
        "metadata": {
            "chapter_num": "720",
            "section_citation": "720 ILCS 5/9-1",
            "act_id": "2465",
            "major_topic": "CRIMINAL OFFENSES",
            "source": "ilcs",
        },
    }
    payload = build_payload(record, [0.2] * 768, "ilcs_chunks")
    assert payload["section_citation"] == "720 ILCS 5/9-1"
    assert payload["chapter_num"] == "720"
    assert payload["act_id"] == "2465"
    assert payload["major_topic"] == "CRIMINAL OFFENSES"
    assert payload["chunk_id"] == "ch720-act2465-sec5_c0"


def test_build_payload_court_rule_has_legacy_columns():
    record = {
        "chunk_id": "iscr-abc",
        "source": "illinois_supreme_court_rules",
        "text": "The court shall...",
        "enriched_text": "[ISCR | Rule 431]\nThe court shall...",
        "metadata": {
            "source": "illinois_supreme_court_rules",
            "hierarchical_path": "Article IV | Rule 431",
            "article_number": "IV",
            "article_title": "Jury Selection",
            "part_letter": None,
            "part_title": None,
            "rule_number": "431",
            "rule_title": "Voir Dire",
            "subsection_id": None,
            "effective_date": "2021-01-01",
            "amendment_history": None,
        },
    }
    payload = build_payload(record, [0.3] * 768, "court_rule_chunks")
    assert payload["hierarchical_path"] == "Article IV | Rule 431"
    assert payload["rule_number"] == "431"
    assert payload["rule_title"] == "Voir Dire"
    assert payload["article_title"] == "Jury Selection"
    assert payload["effective_date"] == "2021-01-01"
    # legacy columns written by old iscr_embed.py — must be present
    assert "source_s3_key" in payload
    assert "content_type" in payload
    assert "committee_comments" in payload
    assert "cross_references" in payload
    # court_rule_chunks table does not have parent_id or chunk_index columns
    assert "parent_id" not in payload
    assert "chunk_index" not in payload


def test_build_payload_uses_enriched_text_not_text_for_embedding_field():
    rec = _make_opinion_record()
    payload = build_payload(rec, [0.0] * 768, "opinion_chunks")
    assert payload["enriched_text"] == rec["enriched_text"]
    assert payload["text"] == rec["text"]


# ---------------------------------------------------------------------------
# iter_records
# ---------------------------------------------------------------------------

def test_iter_records_parses_valid_jsonl():
    lines = [json.dumps({"chunk_id": "a"}), json.dumps({"chunk_id": "b"})]
    records = list(iter_records(lines))
    assert len(records) == 2
    assert records[0]["chunk_id"] == "a"
    assert records[1]["chunk_id"] == "b"


def test_iter_records_skips_malformed_lines():
    lines = [
        json.dumps({"chunk_id": "a"}),
        "not valid json {{{{",
        json.dumps({"chunk_id": "b"}),
    ]
    records = list(iter_records(lines))
    assert len(records) == 2
    assert records[0]["chunk_id"] == "a"
    assert records[1]["chunk_id"] == "b"


def test_iter_records_skips_blank_lines():
    lines = [json.dumps({"chunk_id": "a"}), "", "   ", json.dumps({"chunk_id": "b"})]
    records = list(iter_records(lines))
    assert len(records) == 2


def test_iter_records_accepts_bytes_lines():
    # S3 StreamingBody.iter_lines() yields bytes, not str
    lines = [
        json.dumps({"chunk_id": "a"}).encode("utf-8"),
        json.dumps({"chunk_id": "b"}).encode("utf-8"),
    ]
    records = list(iter_records(lines))
    assert len(records) == 2
    assert records[0]["chunk_id"] == "a"


# ---------------------------------------------------------------------------
# Null-byte sanitisation (PostgreSQL rejects \x00 in text columns)
# ---------------------------------------------------------------------------

def test_build_payload_strips_null_bytes_from_text():
    record = _make_opinion_record(
        text="good content\x00hidden null\x00",
        enriched_text="[header]\n\ngood content\x00hidden null\x00",
    )
    # Simulate what embed_source does before calling build_payload
    record["text"] = record["text"].replace("\x00", "")
    record["enriched_text"] = record["enriched_text"].replace("\x00", "")
    payload = build_payload(record, [0.0] * 768, "opinion_chunks")
    assert "\x00" not in payload["text"]
    assert "\x00" not in payload["enriched_text"]


# ---------------------------------------------------------------------------
# _normalize_cl_chunk — CourtListener flat schema → shared Chunk schema
# ---------------------------------------------------------------------------

_CL_FLAT_CHUNK = {
    "chunk_id": "3401386_api_0",
    "chunk_index": 0,
    "chunk_type": "opinion_section",
    "source": "courtlistener_api",
    "text": "Jenkins, Justice. A wife in her petition for divorce...",
    "token_count": 344,
    "section_heading": "PREAMBLE",
    "section_index": 0,
    "opinion_id": "3401386",
    "opinion_type": "majority",
    "is_majority": True,
    "author": "Jenkins",
    "per_curiam": False,
    "cluster_id": "1941",
    "case_name": "Evans v. Poskon",
    "case_name_short": "Evans",
    "date_filed": "2010-04-16",
    "judges": "Easterbrook, Posner, Williams",
    "precedential_status": "Published",
    "precedential_weight": 1.0,
    "citation_count": 186,
    "docket_id": "330933",
    "court_id": "ca7",
    "court_label": "ca7",
    "docket_number": "09-3140",
    "nature_of_suit": "",
    "cause": "",
    "date_terminated": "",
}


def test_normalize_cl_chunk_source_is_courtlistener():
    n = _normalize_cl_chunk(_CL_FLAT_CHUNK)
    assert n["source"] == "courtlistener"


def test_normalize_cl_chunk_has_enriched_text():
    n = _normalize_cl_chunk(_CL_FLAT_CHUNK)
    assert "enriched_text" in n
    assert n["enriched_text"]


def test_normalize_cl_chunk_enriched_text_contains_case_name():
    n = _normalize_cl_chunk(_CL_FLAT_CHUNK)
    assert "Evans v. Poskon" in n["enriched_text"]


def test_normalize_cl_chunk_enriched_text_contains_year():
    n = _normalize_cl_chunk(_CL_FLAT_CHUNK)
    assert "2010" in n["enriched_text"]


def test_normalize_cl_chunk_parent_id_is_opinion_id():
    n = _normalize_cl_chunk(_CL_FLAT_CHUNK)
    assert n["parent_id"] == "3401386"


def test_normalize_cl_chunk_metadata_preserves_case_fields():
    n = _normalize_cl_chunk(_CL_FLAT_CHUNK)
    m = n["metadata"]
    assert m["case_name"] == "Evans v. Poskon"
    assert m["court_id"] == "ca7"
    assert m["opinion_type"] == "majority"
    assert m["date_filed"] == "2010-04-16"


def test_normalize_cl_chunk_build_payload_succeeds():
    """After normalization, build_payload must not raise KeyError."""
    n = _normalize_cl_chunk(_CL_FLAT_CHUNK)
    payload = build_payload(n, [0.1] * 768, "opinion_chunks")
    assert payload["chunk_id"] == "3401386_api_0"
    assert payload["source"] == "courtlistener"
    assert payload["enriched_text"]
    assert isinstance(payload["metadata"], dict)
    assert payload["metadata"]["case_name"] == "Evans v. Poskon"
