import argparse
import json
import os
import logging
import sys
import boto3
from pathlib import Path
from typing import Generator
from dotenv import load_dotenv
import re
from supabase import create_client, Client

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))
from retrieval.embeddings import get_embedding_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

BATCH_SIZE = 200
MAX_EMBED_CHARS = 1500  # mxbai-embed-large is BERT-based (512-token limit); legal text tokenizes
                        # at ~3 chars/token, so 1500 chars ≈ 500 tokens — safely within the limit


def _checkpoint_file(table: str) -> str:
    return f"data_files/iscr_embedded_chunks_{table}.txt"


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


def load_checkpoint(supabase: Client, table: str) -> set[str]:
    db_ids: set[str] = set()
    page_size = 1000
    offset = 0
    while True:
        rows = (
            supabase.table(table)
            .select("chunk_id")
            .range(offset, offset + page_size - 1)
            .execute()
            .data
        )
        db_ids.update(r["chunk_id"] for r in rows)
        if len(rows) < page_size:
            break
        offset += page_size
    log.info("Found %d chunks already in %s.", len(db_ids), table)

    checkpoint = _checkpoint_file(table)
    if db_ids:
        with open(checkpoint, "w") as f:
            for chunk_id in sorted(db_ids):
                f.write(chunk_id + "\n")

    return db_ids


def save_checkpoint(chunk_ids: list[str], table: str) -> None:
    with open(_checkpoint_file(table), "a") as f:
        for chunk_id in chunk_ids:
            f.write(chunk_id + "\n")


def flush_batch(supabase: Client, batch: list[dict], table: str) -> list[str]:
    if not batch:
        return []
    chunk_ids = [p["chunk_id"] for p in batch]
    try:
        supabase.table(table).upsert(batch).execute()
        log.info("Flushed %d chunks to %s.", len(batch), table)
        return chunk_ids
    except Exception as e:
        log.error("Failed to flush batch of %d chunks: %s", len(batch), e)
        return []


def run(local_input: Path | None = None, table: str = "court_rule_chunks") -> None:
    cfg = get_config()
    supabase: Client = create_client(cfg["supabase_url"], cfg["supabase_key"])
    embed_model = get_embedding_model()
    log.info("Embedding into table %r using model %r", table, embed_model.__class__.__name__)

    if local_input:
        log.info("Reading chunks from local file: %s", local_input)
        data = local_input.read_text(encoding="utf-8")
    else:
        data = read_raw_corpus_s3(cfg["chunked_bucket"], cfg["chunked_key"], cfg["aws_region"])

    processed = load_checkpoint(supabase, table)
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
        if len(enriched_text) > MAX_EMBED_CHARS:
            log.warning("Chunk %s exceeded max length (%d chars), truncating.", chunk_id, len(enriched_text))
            enriched_text = enriched_text[:MAX_EMBED_CHARS]
        embedding = embed_model.get_text_embedding(enriched_text)
        payload = build_payload(record, embedding, enriched_text)
        batch.append(payload)
        total += 1
        if len(batch) >= BATCH_SIZE:
            flushed_ids = flush_batch(supabase, batch, table)
            save_checkpoint(flushed_ids, table)
            processed.update(flushed_ids)
            batch = []

    if batch:
        flushed_ids = flush_batch(supabase, batch, table)
        save_checkpoint(flushed_ids, table)

    log.info("Done. Embedded %d chunks, skipped %d already processed.", total, skipped)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed Illinois Supreme Court Rules chunks into Supabase"
    )
    parser.add_argument(
        "--table",
        default="court_rule_chunks",
        metavar="TABLE",
        help="Target Supabase table name (default: court_rule_chunks).",
    )
    args = parser.parse_args()
    run(table=args.table)


if __name__ == "__main__":
    main()
