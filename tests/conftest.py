import io
import json
from pathlib import Path

import boto3
import pdfplumber
import pytest
from dotenv import load_dotenv
import os

load_dotenv()

FIXTURES_DIR = Path(__file__).parent / "fixtures"
ISCR_FIXTURES_DIR = FIXTURES_DIR / "iscr"


@pytest.fixture(scope="session")
def iscr_texts():
    """Download ISCR PDFs from S3 once, extract text with pdfplumber, cache as .txt files."""
    ISCR_FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    bucket = os.environ["RAW_S3_BUCKET"]
    prefix = os.environ["SUPREME_COURT_RULES_S3_PREFIX"].rstrip("/")
    s3 = boto3.client("s3")

    paginator = s3.get_paginator("list_objects_v2")
    pdf_keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/"):
        for obj in page.get("Contents", []):
            if obj["Key"].lower().endswith(".pdf"):
                pdf_keys.append(obj["Key"])

    if not pdf_keys:
        pytest.skip(f"No PDFs found in s3://{bucket}/{prefix}/ — check credentials and S3 prefix")

    texts = {}
    for key in pdf_keys:
        stem = key[len(prefix):].lstrip("/").replace("/", "_").removesuffix(".pdf")
        cache_path = ISCR_FIXTURES_DIR / f"{stem}.txt"
        if cache_path.exists():
            texts[stem] = cache_path.read_text(encoding="utf-8")
        else:
            response = s3.get_object(Bucket=bucket, Key=key)
            pdf_bytes = response["Body"].read()
            page_texts = []
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text(x_tolerance=2, y_tolerance=2)
                    if text:
                        page_texts.append(f"[PAGE {i}]\n{text.strip()}")
            full_text = "\n\n".join(page_texts)
            cache_path.write_text(full_text, encoding="utf-8")
            texts[stem] = full_text

    return texts


@pytest.fixture(scope="session")
def ilcs_records():
    """Load ilcs_corpus.jsonl — from data_files/corpus/ if present, else download from S3."""
    local_path = Path(__file__).parent.parent / "data_files" / "corpus" / "ilcs_corpus.jsonl"
    if not local_path.exists():
        bucket = os.environ.get("RAW_S3_BUCKET")
        s3_key = os.environ.get("ILCS_S3_PREFIX", "ilcs/").rstrip("/") + "/ilcs_corpus.jsonl"
        if not bucket:
            pytest.skip("ilcs_corpus.jsonl not found locally and RAW_S3_BUCKET not set")
        s3 = boto3.client("s3")
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, s3_key, str(local_path))
        except Exception as e:
            pytest.skip(f"Could not download s3://{bucket}/{s3_key}: {e}")
    records = []
    with open(local_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@pytest.fixture(scope="session")
def ilcs_chunks(ilcs_records):
    """Run chunk_section() over all ilcs_records; return flat list of all chunks."""
    from chunk.ilga_chunk import chunk_section
    chunks = []
    for rec in ilcs_records:
        chunks.extend(chunk_section(rec))
    return chunks


@pytest.fixture(scope="session")
def ilcs_chunks_s3():
    """Load the actual chunked JSONL from the chunked S3 bucket."""
    bucket = os.environ.get("CHUNKED_S3_BUCKET")
    if not bucket:
        pytest.skip("CHUNKED_S3_BUCKET not set")
    key = os.environ.get("ILCS_CHUNKED_OBJECT_KEY",
                         os.environ.get("ILCS_S3_PREFIX", "ilcs").rstrip("/") + "/ilcs_chunks.jsonl")
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        raw = obj["Body"].read().decode("utf-8")
    except Exception as e:
        pytest.skip(f"Could not download {key} from S3: {e}")
    chunks = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            chunks.append(json.loads(line))
    return chunks


@pytest.fixture(scope="session")
def iscr_chunks(iscr_texts):
    """Run chunk_document() over all ISCR texts; return flat list of all chunks."""
    from chunk.iscr_chunk import chunk_document
    chunks = []
    for stem, text in iscr_texts.items():
        chunks.extend(chunk_document(text, source_key=f"{stem}.pdf"))
    return chunks


@pytest.fixture(scope="session")
def iscr_chunks_s3():
    """Load all ISCR chunked JSONLs from the chunked S3 bucket."""
    bucket = os.environ.get("CHUNKED_S3_BUCKET")
    prefix = os.environ.get("SUPREME_COURT_RULES_S3_PREFIX", "illinois-supreme-court-rules").rstrip("/")
    if not bucket:
        pytest.skip("CHUNKED_S3_BUCKET not set")
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    chunk_keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith("_chunks.jsonl"):
                chunk_keys.append(obj["Key"])
    if not chunk_keys:
        pytest.skip(f"No _chunks.jsonl files found in s3://{bucket}/{prefix}/")
    chunks = []
    for key in chunk_keys:
        obj = s3.get_object(Bucket=bucket, Key=key)
        for line in obj["Body"].read().decode("utf-8").splitlines():
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


@pytest.fixture(scope="session")
def idoc_records():
    """Download idoc_corpus.jsonl from S3 and return parsed records."""
    bucket = os.environ.get("RAW_S3_BUCKET")
    if not bucket:
        pytest.skip("RAW_S3_BUCKET not set")
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key="idoc/idoc_corpus.jsonl")
        raw = obj["Body"].read().decode("utf-8")
    except Exception as e:
        pytest.skip(f"Could not download idoc_corpus.jsonl from S3: {e}")
    return [json.loads(l) for l in raw.splitlines() if l.strip()]


@pytest.fixture(scope="session")
def idoc_chunks(idoc_records):
    """Run chunk_record() over de-duplicated IDOC records; return flat list of all chunks."""
    from chunk.idoc_chunk import chunk_record, deduplicate_records
    from dataclasses import asdict
    chunks = []
    for rec in deduplicate_records(idoc_records):
        chunks.extend(asdict(c) for c in chunk_record(rec))
    return chunks


@pytest.fixture(scope="session")
def idoc_chunks_s3():
    """Load the actual chunked JSONL from the chunked S3 bucket."""
    bucket = os.environ.get("CHUNKED_S3_BUCKET")
    if not bucket:
        pytest.skip("CHUNKED_S3_BUCKET not set")
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key="idoc/idoc_chunks.jsonl")
        raw = obj["Body"].read().decode("utf-8")
    except Exception as e:
        pytest.skip(f"Could not download idoc_chunks.jsonl from S3: {e}")
    return [json.loads(l) for l in raw.splitlines() if l.strip()]


@pytest.fixture(scope="session")
def iac_records():
    """Download iac_corpus.jsonl from S3 and return parsed records."""
    bucket = os.environ.get("RAW_S3_BUCKET")
    if not bucket:
        pytest.skip("RAW_S3_BUCKET not set")
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key="iac/iac_corpus.jsonl")
        raw = obj["Body"].read().decode("utf-8")
    except Exception as e:
        pytest.skip(f"Could not download iac_corpus.jsonl from S3: {e}")
    records = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


@pytest.fixture(scope="session")
def iac_chunks(iac_records):
    """Run chunk_section() over de-duplicated IAC records; return flat list of all chunks."""
    from chunk.iac_chunk import chunk_section, deduplicate_records
    chunks = []
    for rec in deduplicate_records(iac_records):
        chunks.extend(chunk_section(rec))
    return chunks


@pytest.fixture(scope="session")
def iac_chunks_s3():
    """Load the actual chunked JSONL from the chunked S3 bucket."""
    bucket = os.environ.get("CHUNKED_S3_BUCKET")
    if not bucket:
        pytest.skip("CHUNKED_S3_BUCKET not set")
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key="iac/iac_chunks.jsonl")
        raw = obj["Body"].read().decode("utf-8")
    except Exception as e:
        pytest.skip(f"Could not download iac_chunks.jsonl from S3: {e}")
    chunks = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            chunks.append(json.loads(line))
    return chunks


@pytest.fixture(scope="session")
def spac_records():
    """Download spac_corpus.jsonl from S3 and return parsed records."""
    bucket = os.environ.get("RAW_S3_BUCKET")
    if not bucket:
        pytest.skip("RAW_S3_BUCKET not set")
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key="spac/spac_corpus.jsonl")
        raw = obj["Body"].read().decode("utf-8")
    except Exception as e:
        pytest.skip(f"Could not download spac_corpus.jsonl from S3: {e}")
    return [json.loads(l) for l in raw.splitlines() if l.strip()]


@pytest.fixture(scope="session")
def spac_chunks(spac_records):
    """Run chunk_record() over de-duplicated SPAC records; return flat list of all chunks as dicts."""
    from chunk.spac_chunk import chunk_record, deduplicate_records
    from dataclasses import asdict
    chunks = []
    for rec in deduplicate_records(spac_records):
        chunks.extend(asdict(c) for c in chunk_record(rec))
    return chunks


@pytest.fixture(scope="session")
def spac_chunks_s3():
    """Load the actual chunked JSONL from the chunked S3 bucket."""
    bucket = os.environ.get("CHUNKED_S3_BUCKET")
    if not bucket:
        pytest.skip("CHUNKED_S3_BUCKET not set")
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key="spac/spac_chunks.jsonl")
        raw = obj["Body"].read().decode("utf-8")
    except Exception as e:
        pytest.skip(f"Could not download spac_chunks.jsonl from S3: {e}")
    return [json.loads(l) for l in raw.splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# ICCB fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def iccb_records():
    """Download iccb_corpus.jsonl from S3 and return parsed records."""
    bucket = os.environ.get("RAW_S3_BUCKET")
    if not bucket:
        pytest.skip("RAW_S3_BUCKET not set")
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key="iccb/iccb_corpus.jsonl")
        raw = obj["Body"].read().decode("utf-8")
    except Exception as e:
        pytest.skip(f"Could not download iccb_corpus.jsonl from S3: {e}")
    return [json.loads(l) for l in raw.splitlines() if l.strip()]


@pytest.fixture(scope="session")
def iccb_chunks(iccb_records):
    """Run chunk_record() over de-duplicated ICCB records; return flat list of all chunks."""
    from chunk.iccb_chunk import chunk_record, deduplicate_records
    from dataclasses import asdict
    chunks = []
    for rec in deduplicate_records(iccb_records):
        chunks.extend(asdict(c) for c in chunk_record(rec))
    return chunks


@pytest.fixture(scope="session")
def iccb_chunks_s3():
    """Load the actual chunked JSONL from the chunked S3 bucket."""
    bucket = os.environ.get("CHUNKED_S3_BUCKET")
    if not bucket:
        pytest.skip("CHUNKED_S3_BUCKET not set")
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key="iccb/iccb_chunks.jsonl")
        raw = obj["Body"].read().decode("utf-8")
    except Exception as e:
        pytest.skip(f"Could not download iccb_chunks.jsonl from S3: {e}")
    return [json.loads(l) for l in raw.splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Federal fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def federal_records():
    """Download federal_corpus.jsonl from S3 and return parsed records."""
    bucket = os.environ.get("RAW_S3_BUCKET")
    if not bucket:
        pytest.skip("RAW_S3_BUCKET not set")
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key="federal/federal_corpus.jsonl")
        raw = obj["Body"].read().decode("utf-8")
    except Exception as e:
        pytest.skip(f"Could not download federal_corpus.jsonl from S3: {e}")
    return [json.loads(l) for l in raw.splitlines() if l.strip()]


@pytest.fixture(scope="session")
def federal_chunks(federal_records):
    """Run chunk_record() over de-duplicated federal records; return flat list of all chunks."""
    from chunk.federal_chunk import chunk_record, deduplicate_records
    from dataclasses import asdict
    chunks = []
    for rec in deduplicate_records(federal_records):
        chunks.extend(asdict(c) for c in chunk_record(rec))
    return chunks


@pytest.fixture(scope="session")
def federal_chunks_s3():
    """Load the actual chunked JSONL from the chunked S3 bucket."""
    bucket = os.environ.get("CHUNKED_S3_BUCKET")
    if not bucket:
        pytest.skip("CHUNKED_S3_BUCKET not set")
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key="federal/federal_chunks.jsonl")
        raw = obj["Body"].read().decode("utf-8")
    except Exception as e:
        pytest.skip(f"Could not download federal_chunks.jsonl from S3: {e}")
    return [json.loads(l) for l in raw.splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Restore Justice fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def restorejustice_records():
    """Download restorejustice_corpus.jsonl from S3 and return parsed records."""
    bucket = os.environ.get("RAW_S3_BUCKET")
    if not bucket:
        pytest.skip("RAW_S3_BUCKET not set")
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key="restorejustice/restorejustice_corpus.jsonl")
        raw = obj["Body"].read().decode("utf-8")
    except Exception as e:
        pytest.skip(f"Could not download restorejustice_corpus.jsonl from S3: {e}")
    return [json.loads(l) for l in raw.splitlines() if l.strip()]


@pytest.fixture(scope="session")
def restorejustice_chunks(restorejustice_records):
    """Run chunk_record() over de-duplicated RJ records; return flat list of all chunks."""
    from chunk.restorejustice_chunk import chunk_record, deduplicate_records
    from dataclasses import asdict
    chunks = []
    for rec in deduplicate_records(restorejustice_records):
        chunks.extend(asdict(c) for c in chunk_record(rec))
    return chunks


@pytest.fixture(scope="session")
def restorejustice_chunks_s3():
    """Load the actual chunked JSONL from the chunked S3 bucket."""
    bucket = os.environ.get("CHUNKED_S3_BUCKET")
    if not bucket:
        pytest.skip("CHUNKED_S3_BUCKET not set")
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key="restorejustice/restorejustice_chunks.jsonl")
        raw = obj["Body"].read().decode("utf-8")
    except Exception as e:
        pytest.skip(f"Could not download restorejustice_chunks.jsonl from S3: {e}")
    return [json.loads(l) for l in raw.splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Cook County PD fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def cookcounty_pd_records():
    """Download cookcounty_pd_corpus.jsonl from S3 and return parsed records."""
    bucket = os.environ.get("RAW_S3_BUCKET")
    if not bucket:
        pytest.skip("RAW_S3_BUCKET not set")
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key="cookcounty-pd/cookcounty_pd_corpus.jsonl")
        raw = obj["Body"].read().decode("utf-8")
    except Exception as e:
        pytest.skip(f"Could not download cookcounty_pd_corpus.jsonl from S3: {e}")
    return [json.loads(l) for l in raw.splitlines() if l.strip()]


@pytest.fixture(scope="session")
def cookcounty_pd_chunks(cookcounty_pd_records):
    """Run chunk_record() over de-duplicated Cook County PD records; return flat list."""
    from chunk.cookcounty_pd_chunk import chunk_record, deduplicate_records
    from dataclasses import asdict
    chunks = []
    for rec in deduplicate_records(cookcounty_pd_records):
        chunks.extend(asdict(c) for c in chunk_record(rec))
    return chunks


@pytest.fixture(scope="session")
def cookcounty_pd_chunks_s3():
    """Load the actual chunked JSONL from the chunked S3 bucket."""
    bucket = os.environ.get("CHUNKED_S3_BUCKET")
    if not bucket:
        pytest.skip("CHUNKED_S3_BUCKET not set")
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key="cookcounty-pd/cookcounty_pd_chunks.jsonl")
        raw = obj["Body"].read().decode("utf-8")
    except Exception as e:
        pytest.skip(f"Could not download cookcounty_pd_chunks.jsonl from S3: {e}")
    return [json.loads(l) for l in raw.splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# CourtListener fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def cl_raw_dfs():
    """Download CL bulk CSVs from S3; return (clusters_df, dockets_df, opinions_df, parentheticals_df)."""
    import io
    import pandas as pd
    bucket = os.environ.get("RAW_S3_BUCKET")
    if not bucket:
        pytest.skip("RAW_S3_BUCKET not set")
    cl_prefix = os.environ.get("RAW_COURTLISTENER_S3_PREFIX", "courtlistener").rstrip("/")
    prefix = f"{cl_prefix}/bulk"
    s3 = boto3.client("s3")
    dfs = {}
    for name in ("clusters", "dockets", "opinions", "parentheticals"):
        key = f"{prefix}/{name}.csv"
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            dfs[name] = pd.read_csv(
                io.BytesIO(obj["Body"].read()),
                low_memory=False,
                dtype=str,
                keep_default_na=False,
            )
        except Exception as e:
            pytest.skip(f"Could not download s3://{bucket}/{key}: {e}")
    return dfs["clusters"], dfs["dockets"], dfs["opinions"], dfs["parentheticals"]


@pytest.fixture(scope="session")
def cl_opinion_chunks(cl_raw_dfs):
    """Run chunk_opinion() over all CL opinions; return flat list of chunks as dicts."""
    import dataclasses
    from chunk.courtlistener_chunk import (
        build_lookup_maps, chunk_opinion, safe_str, MIN_CHUNK_TOKENS,
    )
    clusters_df, dockets_df, opinions_df, _ = cl_raw_dfs
    cluster_map, docket_map = build_lookup_maps(clusters_df, dockets_df)
    chunks = []
    for _, row in opinions_df.iterrows():
        opinion_chunks = chunk_opinion(row, cluster_map, docket_map)
        chunks.extend(dataclasses.asdict(c) for c in opinion_chunks if c.token_count >= MIN_CHUNK_TOKENS)
    if not chunks:
        pytest.skip("No CL opinion chunks produced — corpus may be empty")
    return chunks


@pytest.fixture(scope="session")
def cl_opinion_chunks_s3():
    """Load the actual opinion_chunks.jsonl from the chunked S3 bucket."""
    bucket = os.environ.get("CHUNKED_S3_BUCKET")
    if not bucket:
        pytest.skip("CHUNKED_S3_BUCKET not set")
    cl_prefix = os.environ.get("RAW_COURTLISTENER_S3_PREFIX", "courtlistener").rstrip("/")
    key = f"{cl_prefix}/bulk/opinion_chunks.jsonl"
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        raw = obj["Body"].read().decode("utf-8")
    except Exception as e:
        pytest.skip(f"Could not download {key} from S3: {e}")
    return [json.loads(l) for l in raw.splitlines() if l.strip()]
