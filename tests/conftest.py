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

    texts = {}
    for key in pdf_keys:
        stem = Path(key).stem
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
    """Load all records from the local ilcs_corpus.jsonl."""
    corpus_path = Path(__file__).parent.parent / "ilcs_corpus.jsonl"
    records = []
    with open(corpus_path, encoding="utf-8") as f:
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
def iscr_chunks(iscr_texts):
    """Run chunk_document() over all ISCR texts; return flat list of all chunks."""
    from chunk.iscr_chunk import chunk_document
    chunks = []
    for stem, text in iscr_texts.items():
        chunks.extend(chunk_document(text, source_key=f"{stem}.pdf"))
    return chunks
