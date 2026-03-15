import json
import os
import logging
import sys
import boto3
from typing import Generator
from dotenv import load_dotenv
import re
from llama_index.embeddings.ollama import OllamaEmbedding
from supabase import create_client, Client

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
    rules_prefix = os.getenv("SUPREME_COURT_RULES_S3_PREFIX", "illinois-supreme-court-rules").rstrip("/")
    return {
        "chunked_bucket": _require_env("CHUNKED_S3_BUCKET"),
        "chunked_key":    os.getenv("ISCR_CHUNKED_OBJECT_KEY", f"{rules_prefix}/S_Ct_Rules_full_chunks.jsonl"),
        "aws_region":     os.getenv("AWS_REGION"),
        "supabase_url":   _require_env("SUPABASE_URL"),
        "supabase_key":   _require_env("SUPABASE_SERVICE_KEY"),
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

def clean_chunk_text(text: str) -> str:
    text = re.sub(r'\[PAGE\s+\d+\]', '', text)
    text = re.sub(r'\n(?:Amended|Adopted|Effective)\s+[A-Za-z]+\s+\d{1,2},\s+\d{4}[^\n]*', '', text)
    return re.sub(r'\n{3,}', '\n\n', text).strip()

def build_chunk(record: dict) -> str:
    parts = ["[Illinois Supreme Court Rules"]
    if record.get("article_title"):
        parts[0] += f" | Article {record['article_number']}: {record['article_title']}"
    if record.get("part_title"):
        parts[0] += f" | Part {record['part_letter']}: {record['part_title']}"
    parts[0] += "]"
    header = parts[0] + "\n"
    if record.get("rule_number") and record.get("rule_title"):
        header += f"Rule {record['rule_number']}: {record['rule_title']}"
        if record.get("subsection_id") and record["subsection_id"] != "intro":
            header += f" ({record['subsection_id']})"
        header += "\n"
    if record.get("effective_date"):
        header += f"Effective: {record['effective_date']}\n"
    return header + "\n" + record["text"]

def build_payload(record: dict, embedding: list[float], enriched_text: str) -> dict:
    return {
        "chunk_id":           record["chunk_id"],
        "source":             record.get("source_corpus"),
        "source_s3_key":      record.get("source_s3_key"),
        "content_type":       record.get("content_type"),
        "hierarchical_path":  record.get("hierarchical_path"),
        "article_number":     record.get("article_number"),
        "article_title":      record.get("article_title"),
        "part_letter":        record.get("part_letter"),
        "part_title":         record.get("part_title"),
        "rule_number":        record.get("rule_number"),
        "rule_title":         record.get("rule_title"),
        "subsection_id":      record.get("subsection_id"),
        "effective_date":     record.get("effective_date"),
        "amendment_history":  record.get("amendment_history"),
        "committee_comments": record.get("committee_comments"),
        "cross_references":   record.get("cross_references"),
        "text":               record["text"],
        "enriched_text":      enriched_text,
        "embedding":          embedding,
    }

def load_checkpoint() -> set[str]:
    if not os.path.exists(CHECKPOINT_FILE):
        log.info("No checkpoint file found, starting fresh.")
        return set()
    with open(CHECKPOINT_FILE, "r") as f:
        ids = {line.strip() for line in f if line.strip()}
    log.info("Loaded %d processed chunk IDs from checkpoint.", len(ids))
    return ids

def save_checkpoint(chunk_ids: list[str]) -> None:
    with open(CHECKPOINT_FILE, "a") as f:
        for chunk_id in chunk_ids:
            f.write(chunk_id + "\n")

def flush_batch(supabase: Client, batch: list[dict]) -> list[str]:
    if not batch:
        return []
    chunk_ids = [p["chunk_id"] for p in batch]
    try:
        supabase.table("court_rule_chunks").upsert(batch).execute()
        log.info("Flushed %d chunks to Supabase.", len(batch))
        return chunk_ids
    except Exception as e:
        log.error("Failed to flush batch of %d chunks: %s", len(batch), e)
        return []

embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
)

CHECKPOINT_FILE = "data_files/iscr_embedded_chunks.txt"
BATCH_SIZE = 200

cfg = get_config()
data = read_raw_corpus_s3(
    bucket=cfg["chunked_bucket"],
    key=cfg["chunked_key"],
    region=cfg["aws_region"]
)
supabase: Client = create_client(cfg["supabase_url"], cfg["supabase_key"])

processed = load_checkpoint()
batch = []
skipped = 0
total = 0

for record in iter_records(data):
    chunk_id = record["chunk_id"]
    if chunk_id in processed:
        skipped += 1
        continue
    cleaned_text = clean_chunk_text(record["text"])
    record["text"] = cleaned_text
    enriched_text = build_chunk(record)
    embed_text = enriched_text[:8000]
    if len(enriched_text) > 8000:
        log.warning("Chunk %s exceeded max length (%d chars), truncating for embedding.", chunk_id, len(enriched_text))
    embedding = embed_model.get_text_embedding(embed_text)
    payload = build_payload(record, embedding, enriched_text)
    batch.append(payload)
    total += 1
    if len(batch) >= BATCH_SIZE:
        flushed_ids = flush_batch(supabase, batch)
        save_checkpoint(flushed_ids)
        processed.update(flushed_ids)
        batch = []
if batch:
    flushed_ids = flush_batch(supabase, batch)
    save_checkpoint(flushed_ids)
log.info("Done. Embedded %d chunks, skipped %d already processed.", total, skipped)