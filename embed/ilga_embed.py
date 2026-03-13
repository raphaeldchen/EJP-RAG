import json
import os
import logging
import sys
import boto3
from typing import Generator
from dotenv import load_dotenv
import re
from llama_index.embeddings.ollama import OllamaEmbedding

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

def _require_env(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        log.error(f"Required environment variable {key!r} is not set.")
        sys.exit(1)
    return val

def get_config() -> dict:
    ilcs_prefix = os.getenv("ILCS_S3_PREFIX", "ilcs/").rstrip("/")
    return {
        "chunked_bucket":_require_env("CHUNKED_S3_BUCKET"),
        "chunked_key":os.getenv("ILCS_CHUNKED_OBJECT_KEY", f"{ilcs_prefix}/ilcs_chunks.jsonl"),
        "aws_region":os.getenv("AWS_REGION"),
    }

def read_raw_corpus_s3(bucket: str, key: str, region: str | None) -> str:
    log.info("Reading raw corpus from s3://%s/%s", bucket, key)
    kwargs = {"region_name": region} if region else {}
    obj = boto3.client("s3", **kwargs).get_object(Bucket=bucket, Key=key)
    return obj["Body"].read().decode("utf-8")

def iter_records(source: str) -> Generator[dict, None, None]:
    for line_no, line in enumerate(source.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            log.warning("Skipping malformed line %d: %s", line_no, exc)

def build_chunk(chunk: dict) -> str:
    m = chunk["metadata"]
    header = (
        f"[{m['chapter_name']} | {m['act_name']} | {m['article_name']}]\n"
        f"Section {m['section_citation']}: {m['section_heading']}\n\n"
    )
    return header + chunk["text"]

def clean_chunk_text(text: str) -> str:
    text = re.sub(r'\(from Ch\.\s+\d+[^)]*\)', '', text)
    text = re.sub(r'\(Source:[^)]*\)', '', text)
    return re.sub(r'\n{3,}', '\n\n', text).strip()

embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",  # mxbai-embed-large for better quality embedding (slower)
    base_url="http://localhost:11434",
)

cfg = get_config()
data = read_raw_corpus_s3(
    bucket=cfg["chunked_bucket"],
    key=cfg["chunked_key"],
    region=cfg["aws_region"]
)

for record in iter_records(data):
    chunk_text = build_chunk(record)
    embedding = embed_model.get_text_embedding(chunk_text)
    # create txt file of processed chunk_id's
    # store embedding and metadata directly in supabase