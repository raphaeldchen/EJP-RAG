import argparse
import json
import os
import logging
import sys
import time
import boto3
from pathlib import Path
from typing import Generator
from dotenv import load_dotenv
import re
from supabase import create_client, Client

sys.path.insert(0, str(Path(__file__).parent.parent))
from retrieval.embeddings import get_embedding_model

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

BATCH_SIZE = 20


def _checkpoint_file(table: str) -> str:
    return f"data_files/ilga_embedded_chunks_{table}.txt"

# ---------------------------------------------------------------------------
# Scope filter — only embed chapters relevant to Illinois criminal justice
# ---------------------------------------------------------------------------

# All chapters with clear criminal justice relevance. Chapter 740 (Torts) is
# the only chapter in the corpus with no meaningful criminal justice content.
_ALLOWED_CHAPTERS = {
    "20",   # Executive agencies — filtered further below by act
    "50",   # Local Government (county jails, sheriff authority)
    "225",  # Professions (licensing consequences of conviction)
    "325",  # Employment (background checks, collateral consequences)
    "410",  # Public Health (drug treatment, sexual assault procedures)
    "430",  # Fire Safety — includes FOID Card Act and Concealed Carry Act
    "625",  # Vehicles (DUI)
    "705",  # Courts / Juvenile Justice
    "720",  # Criminal Offenses
    "725",  # Criminal Procedure
    "730",  # Corrections and Sentencing
    "735",  # Civil Procedure (post-conviction relief, habeas corpus)
    "750",  # Family (domestic violence, orders of protection)
    "775",  # Civil Rights
}

# Chapter 20 is a grab-bag of executive departments. Exclude acts that have no
# connection to criminal justice (Commerce, Natural Resources, Lottery, DoIT,
# Revenue, Investment). Everything else in chapter 20 is kept — DOC, DHS,
# Prisoner Review Board, Expungement, Criminal Justice Information Authority, etc.
_EXCLUDED_CHAPTER_20_ACT_PREFIXES = {
    "20 ILCS 605",   # Department of Commerce and Economic Opportunity
    "20 ILCS 1205",  # Department of Natural Resources
    "20 ILCS 1370",  # Department of Innovation and Technology
    "20 ILCS 1605",  # Illinois Lottery
    "20 ILCS 2505",  # Department of Revenue
    "20 ILCS 3205",  # Investment Officer
}


def _is_in_scope(record: dict) -> bool:
    m = record.get("metadata", {})
    chapter = m.get("chapter_num", "")
    if chapter not in _ALLOWED_CHAPTERS:
        return False
    if chapter == "20":
        citation = m.get("section_citation", "")
        if any(citation.startswith(prefix) for prefix in _EXCLUDED_CHAPTER_20_ACT_PREFIXES):
            return False
    return True
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


def load_checkpoint(supabase: Client, table: str) -> set[str]:
    """
    Returns the set of chunk_ids already present in the target table.

    Uses the DB as the authoritative source so the local checkpoint file
    can't drift out of sync (e.g. after a crash or a run on a different machine).
    The local file is kept as a fast-path cache but is always reconciled with the DB.
    """
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


def flush_batch(supabase: Client, batch: list[dict], table: str) -> list[str]:
    if not batch:
        return []
    try:
        supabase.table(table).upsert(batch).execute()
        log.info("Flushed %d chunks to %s.", len(batch), table)
        return [p["chunk_id"] for p in batch]
    except Exception as e:
        if len(batch) == 1:
            log.error("Failed to flush chunk %s even as a single row: %s", batch[0]["chunk_id"], e)
            return []
        mid = len(batch) // 2
        log.warning("Batch of %d failed (%s), splitting into %d + %d and retrying.",
                    len(batch), e, mid, len(batch) - mid)
        time.sleep(2)
        left  = flush_batch(supabase, batch[:mid], table)
        right = flush_batch(supabase, batch[mid:], table)
        return left + right


def run(local_input: Path | None = None, table: str = "ilcs_chunks") -> None:
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
    out_of_scope = 0
    total = 0
    failed = 0

    for record in iter_records(data):
        chunk_id = record["chunk_id"]
        if chunk_id in processed:
            skipped += 1
            continue
        if not _is_in_scope(record):
            out_of_scope += 1
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
            flushed_ids = flush_batch(supabase, batch, table)
            failed += len(batch) - len(flushed_ids)
            save_checkpoint(flushed_ids, table)
            processed.update(flushed_ids)
            batch = []

    if batch:
        flushed_ids = flush_batch(supabase, batch, table)
        failed += len(batch) - len(flushed_ids)
        save_checkpoint(flushed_ids, table)

    log.info("Done. Embedded %d chunks, skipped %d already processed, %d out-of-scope filtered, %d failed.", total, skipped, out_of_scope, failed)


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
    parser.add_argument(
        "--table",
        default="ilcs_chunks",
        metavar="TABLE",
        help="Target Supabase table name (default: ilcs_chunks). "
             "Use a suffixed name like ilcs_chunks_bge_base when testing a new embedding model.",
    )
    args = parser.parse_args()
    run(local_input=args.local_input, table=args.table)


if __name__ == "__main__":
    main()
