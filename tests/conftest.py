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
    """Run chunk_record() over all SPAC records; return flat list of all chunks as dicts."""
    from chunk.spac_chunk import chunk_record
    from dataclasses import asdict
    chunks = []
    for rec in spac_records:
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
