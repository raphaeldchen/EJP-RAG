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
from llama_index.embeddings.ollama import OllamaEmbedding
from supabase import create_client, Client

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

CHECKPOINT_FILE = "data_files/ilga_embedded_chunks.txt"
BATCH_SIZE = 20
# nomic-embed-text (nomic-bert) hard limit is 2048 tokens; BERT tokenizer runs at ~1.5 chars/token
# on legal text, so 2000 chars ≈ 1333 tokens — safely within the limit
MAX_EMBED_CHARS = 2000


def _require_env(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        log.error(f"Required environment variable {key!r} is not set.")
        sys.exit(1)
    return val


def get_config() -> dict:
    ilcs_prefix = os.getenv("ILCS_S3_PREFIX", "ilcs/").rstrip("/")
    return {
        "chunked_bucket": _require_env("CHUNKED_S3_BUCKET"),
        "chunked_key": os.getenv("ILCS_CHUNKED_OBJECT_KEY", f"{ilcs_prefix}/ilcs_chunks.jsonl"),
        "aws_region": os.getenv("AWS_REGION"),
        "supabase_url": _require_env("SUPABASE_URL"),
        "supabase_key": _require_env("SUPABASE_SERVICE_KEY"),
    }


def read_raw_corpus_s3(bucket: str, key: str, region: str | None) -> str:
    log.info("Reading chunks from s3://%s/%s", bucket, key)
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


def build_payload(record: dict, embedding: list[float], enriched_text: str) -> dict:
    m = record["metadata"]
    return {
        "chunk_id":         record["chunk_id"],
        "parent_id":        record.get("parent_id"),
        "chunk_index":      record.get("chunk_index"),
        "chunk_total":      record.get("chunk_total"),
        "source":           m.get("source"),
        "section_citation": m.get("section_citation"),
        "chapter_num":      m.get("chapter_num"),
        "act_id":           m.get("act_id"),
        "major_topic":      m.get("major_topic"),
        "text":             record["text"],
        "enriched_text":    enriched_text,
        "metadata":         m,
        "embedding":        embedding,
    }


def flush_batch(supabase: Client, batch: list[dict]) -> list[str]:
    if not batch:
        return []
    chunk_ids = [p["chunk_id"] for p in batch]
    try:
        supabase.table("ilcs_chunks").upsert(batch).execute()
        log.info("Flushed %d chunks to Supabase.", len(batch))
        return chunk_ids
    except Exception as e:
        log.error("Failed to flush batch of %d chunks: %s", len(batch), e)
        return []


def run(local_input: Path | None = None) -> None:
    cfg = get_config()
    supabase: Client = create_client(cfg["supabase_url"], cfg["supabase_key"])
    embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",  # mxbai-embed-large for better quality (slower)
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )

    if local_input:
        log.info("Reading chunks from local file: %s", local_input)
        data = local_input.read_text(encoding="utf-8")
    else:
        data = read_raw_corpus_s3(cfg["chunked_bucket"], cfg["chunked_key"], cfg["aws_region"])

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
        if len(enriched_text) > MAX_EMBED_CHARS:
            log.warning("Chunk %s exceeds max embed length (%d chars), truncating.", chunk_id, len(enriched_text))
            enriched_text = enriched_text[:MAX_EMBED_CHARS]
        embedding = embed_model.get_text_embedding(enriched_text)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed ILCS chunks (from S3 or local file) into Supabase"
    )
    parser.add_argument(
        "--local-input",
        type=Path,
        default=None,
        metavar="FILE",
        help="Read chunks from a local JSONL file instead of S3.",
    )
    args = parser.parse_args()
    run(local_input=args.local_input)


if __name__ == "__main__":
    main()
