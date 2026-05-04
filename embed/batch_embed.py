import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Generator, Iterable, NamedTuple

if TYPE_CHECKING:
    from supabase import Client

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BATCH_SIZE = 200
# nomic-embed-text hard limit ~2048 tokens; legal text tokenises at ~1.5 chars/token,
# so 2000 chars ≈ 1333 tokens — safely within the limit for all current embed backends.
MAX_EMBED_CHARS = 2000


# ---------------------------------------------------------------------------
# Source filters
# ---------------------------------------------------------------------------

# Matches case names containing "People" (criminal appellant/respondent) or
# starting with "In re" (juvenile, habeas, SVP commitment).
_CRIMINAL_CASE_RE = re.compile(r"\bPeople\b|^In re\b", re.IGNORECASE)

# Criminal statute chapters most likely to appear in relevant IL opinions.
_CRIMINAL_STATUTE_RE = re.compile(r"\b(705|720|725|730)\s+ILCS\b")

# Opinions before 1973 interpret the pre-1970 IL Constitution and pre-1973
# Unified Code of Corrections (730 ILCS 5) — mostly noise for this corpus.
_CAP_DATE_CUTOFF = "1973-01-01"


def _is_cap_criminal(record: dict) -> bool:
    """Return True if this CAP opinion chunk is relevant to Illinois criminal justice.

    Accepts on case-name signal (People v. / In re) or statute citation in text.
    Rejects civil opinions (torts, contracts, property, family) that have no
    criminal-law content — ~80 % of the raw CAP corpus.
    Rejects opinions decided before 1973 (pre-UCC era).
    """
    date_decided = record.get("metadata", {}).get("date_decided", "")
    if date_decided and date_decided < _CAP_DATE_CUTOFF:
        return False
    case_name = record.get("metadata", {}).get("case_name", "")
    if _CRIMINAL_CASE_RE.search(case_name):
        return True
    text = record.get("text", "") + " " + record.get("enriched_text", "")
    return bool(_CRIMINAL_STATUTE_RE.search(text))


_ALLOWED_ILCS_CHAPTERS = {
    "20",   # Executive agencies (DOC, DHS, Prisoner Review Board, CJIA — filtered below)
    "50",   # Local Government (county jails, sheriff authority)
    "225",  # Professions (licensing consequences of conviction)
    "325",  # Employment (background checks, collateral consequences)
    "410",  # Public Health (drug treatment, sexual-assault procedures)
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

# Chapter 20 is a grab-bag of executive departments. Exclude acts with no
# criminal-justice connection (Commerce, Natural Resources, Lottery, DoIT,
# Revenue, Investment). Everything else in ch. 20 is kept.
_EXCLUDED_CH20_ACT_PREFIXES = {
    "20 ILCS 605",   # Department of Commerce and Economic Opportunity
    "20 ILCS 1205",  # Department of Natural Resources
    "20 ILCS 1370",  # Department of Innovation and Technology
    "20 ILCS 1605",  # Illinois Lottery
    "20 ILCS 2505",  # Department of Revenue
    "20 ILCS 3205",  # Investment Officer
}


def _is_ilcs_in_scope(record: dict) -> bool:
    m = record.get("metadata", {})
    chapter = m.get("chapter_num", "")
    if chapter not in _ALLOWED_ILCS_CHAPTERS:
        return False
    if chapter == "20":
        citation = m.get("section_citation", "")
        return not any(citation.startswith(p) for p in _EXCLUDED_CH20_ACT_PREFIXES)
    return True


# ---------------------------------------------------------------------------
# CourtListener flat-schema normalizer
# ---------------------------------------------------------------------------

def _normalize_cl_chunk(r: dict) -> dict:
    """Convert courtlistener_api flat schema to the shared Chunk schema.

    courtlistener_api.py predates the shared Chunk dataclass and emits a flat
    record where case metadata lives at the top level and enriched_text is
    absent. Normalize at embed time so the generic build_payload path applies.
    """
    case_name = r.get("case_name", "")
    date_filed = r.get("date_filed", "")
    year = date_filed[:4] if len(date_filed) >= 4 else ""
    court_label = r.get("court_label", "ca7")
    heading = r.get("section_heading", "")
    text = r.get("text", "")

    header = f"[{court_label} | {case_name} ({year})]" if case_name else f"[{court_label}]"
    enriched = f"{header}\n[{heading}]\n\n{text}" if heading else f"{header}\n\n{text}"

    return {
        "chunk_id":         r["chunk_id"],
        "parent_id":        r.get("opinion_id"),
        "chunk_index":      r.get("chunk_index"),
        "chunk_total":      None,
        "source":           "courtlistener",
        "token_count":      r.get("token_count"),
        "display_citation": f"{case_name} ({year})" if case_name else "",
        "text":             text,
        "enriched_text":    enriched,
        "metadata": {
            "chunk_type":          r.get("chunk_type"),
            "section_heading":     r.get("section_heading"),
            "section_index":       r.get("section_index"),
            "opinion_id":          r.get("opinion_id"),
            "opinion_type":        r.get("opinion_type"),
            "is_majority":         r.get("is_majority"),
            "author":              r.get("author"),
            "per_curiam":          r.get("per_curiam"),
            "cluster_id":          r.get("cluster_id"),
            "case_name":           case_name,
            "case_name_short":     r.get("case_name_short"),
            "date_filed":          date_filed,
            "judges":              r.get("judges"),
            "precedential_status": r.get("precedential_status"),
            "precedential_weight": r.get("precedential_weight"),
            "citation_count":      r.get("citation_count"),
            "docket_id":           r.get("docket_id"),
            "court_id":            r.get("court_id"),
            "court_label":         court_label,
            "docket_number":       r.get("docket_number"),
        },
    }


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

class SourceEntry(NamedTuple):
    s3_key: str
    table: str
    filter_fn: Callable[[dict], bool] | None


SOURCE_REGISTRY: dict[str, SourceEntry] = {
    "ilcs":           SourceEntry("ilcs/ilcs_chunks.jsonl",                                    "ilcs_chunks",       _is_ilcs_in_scope),
    "iscr":           SourceEntry("illinois-supreme-court-rules/S_Ct_Rules_full_chunks.jsonl", "court_rule_chunks", None),
    "cap_bulk":       SourceEntry("cap/cap_opinion_chunks.jsonl",                              "opinion_chunks",    _is_cap_criminal),
    "courtlistener":  SourceEntry("courtlistener/bulk/api_opinion_chunks.jsonl",               "opinion_chunks",    None),
    "iac":            SourceEntry("iac/iac_chunks.jsonl",                                      "regulation_chunks", None),
    "idoc":           SourceEntry("idoc/idoc_chunks.jsonl",                                    "regulation_chunks", None),
    "spac":           SourceEntry("spac/spac_chunks.jsonl",                                    "document_chunks",   None),
    "iccb":           SourceEntry("iccb/iccb_chunks.jsonl",                                    "document_chunks",   None),
    "federal":        SourceEntry("federal/federal_chunks.jsonl",                              "document_chunks",   None),
    "restorejustice": SourceEntry("restorejustice/restorejustice_chunks.jsonl",                "document_chunks",   None),
    "cookcounty_pd":  SourceEntry("cookcounty-pd/cookcounty_pd_chunks.jsonl",                  "document_chunks",   None),
}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def iter_records(lines: Iterable[str | bytes]) -> Generator[dict, None, None]:
    for line_no, line in enumerate(lines, start=1):
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            log.warning("Skipping malformed line %d: %s", line_no, exc)


def build_payload(record: dict, embedding: list[float], table: str) -> dict:
    """Build the Supabase upsert payload for the given table.

    ilcs_chunks and court_rule_chunks have legacy per-source column schemas.
    All new tables (opinion_chunks, regulation_chunks, document_chunks) use the
    shared Chunk schema with source-specific fields in the metadata JSONB column.
    """
    enriched = record["enriched_text"]
    m = record.get("metadata", {})

    if table == "ilcs_chunks":
        return {
            "chunk_id":         record["chunk_id"],
            "parent_id":        record.get("parent_id"),
            "chunk_index":      record.get("chunk_index"),
            "chunk_total":      record.get("chunk_total"),
            "source":           record.get("source") or m.get("source"),
            "section_citation": m.get("section_citation"),
            "chapter_num":      m.get("chapter_num"),
            "act_id":           m.get("act_id"),
            "major_topic":      m.get("major_topic"),
            "text":             record["text"],
            "enriched_text":    enriched,
            "metadata":         m,
            "embedding":        embedding,
        }

    if table == "court_rule_chunks":
        # This table predates the shared Chunk schema and lacks parent_id /
        # chunk_index / chunk_total / token_count / display_citation columns.
        return {
            "chunk_id":           record["chunk_id"],
            "source":             record.get("source") or m.get("source"),
            "source_s3_key":      m.get("source_s3_key"),
            "content_type":       m.get("content_type"),
            "hierarchical_path":  m.get("hierarchical_path"),
            "article_number":     m.get("article_number"),
            "article_title":      m.get("article_title"),
            "part_letter":        m.get("part_letter"),
            "part_title":         m.get("part_title"),
            "rule_number":        m.get("rule_number"),
            "rule_title":         m.get("rule_title"),
            "subsection_id":      m.get("subsection_id"),
            "effective_date":     m.get("effective_date"),
            "amendment_history":  m.get("amendment_history"),
            "committee_comments": m.get("committee_comments"),
            "cross_references":   m.get("cross_references"),
            "text":               record["text"],
            "enriched_text":      enriched,
            "embedding":          embedding,
        }

    # Shared Chunk schema — opinion_chunks / regulation_chunks / document_chunks
    return {
        "chunk_id":         record["chunk_id"],
        "parent_id":        record.get("parent_id"),
        "chunk_index":      record.get("chunk_index"),
        "chunk_total":      record.get("chunk_total"),
        "source":           record.get("source"),
        "token_count":      record.get("token_count"),
        "display_citation": record.get("display_citation", ""),
        "text":             record["text"],
        "enriched_text":    enriched,
        "metadata":         record.get("metadata", {}),
        "embedding":        embedding,
    }


def load_checkpoint(supabase: "Client", table: str) -> set[str]:
    """Return chunk_ids already present in the target table (DB is authoritative)."""
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
    return db_ids


def flush_batch(supabase: "Client", batch: list[dict], table: str) -> list[str]:
    """Upsert batch; on failure, binary-split and retry. Returns flushed chunk_ids."""
    if not batch:
        return []
    try:
        supabase.table(table).upsert(batch).execute()
        log.info("Flushed %d chunks to %s.", len(batch), table)
        return [p["chunk_id"] for p in batch]
    except Exception as e:
        if len(batch) == 1:
            log.error("Failed to flush chunk %s: %s", batch[0]["chunk_id"], e)
            return []
        mid = len(batch) // 2
        log.warning("Batch of %d failed (%s), splitting %d + %d and retrying.",
                    len(batch), e, mid, len(batch) - mid)
        time.sleep(2)
        return flush_batch(supabase, batch[:mid], table) + flush_batch(supabase, batch[mid:], table)


# ---------------------------------------------------------------------------
# Per-source embed loop
# ---------------------------------------------------------------------------

def _require_env(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        log.error("Required env var %r is not set.", key)
        sys.exit(1)
    return val


def embed_source(
    source_id: str,
    supabase: "Client",
    embed_model,
    chunked_bucket: str,
    aws_region: str | None,
    local_input: Path | None = None,
) -> None:
    entry = SOURCE_REGISTRY[source_id]
    table = entry.table

    if local_input:
        log.info("[%s] Reading from local file: %s", source_id, local_input)
        lines: Iterable = open(local_input, encoding="utf-8")
    else:
        log.info("[%s] Reading from s3://%s/%s", source_id, chunked_bucket, entry.s3_key)
        import boto3
        kwargs = {"region_name": aws_region} if aws_region else {}
        obj = boto3.client("s3", **kwargs).get_object(Bucket=chunked_bucket, Key=entry.s3_key)
        lines = obj["Body"].iter_lines()

    processed = load_checkpoint(supabase, table)
    batch: list[dict] = []
    embedded = skipped = filtered = failed = 0

    for record in iter_records(lines):
        chunk_id = record["chunk_id"]
        if chunk_id in processed:
            skipped += 1
            continue
        if entry.filter_fn and not entry.filter_fn(record):
            filtered += 1
            continue

        if source_id == "courtlistener":
            record = _normalize_cl_chunk(record)

        # Strip null bytes — PostgreSQL text columns reject \x00 (common in
        # PDF-extracted text from certain encodings).
        record["text"] = record["text"].replace("\x00", "")
        enriched = record["enriched_text"].replace("\x00", "")
        if len(enriched) > MAX_EMBED_CHARS:
            log.debug("Chunk %s truncated from %d to %d chars.", chunk_id, len(enriched), MAX_EMBED_CHARS)
            enriched = enriched[:MAX_EMBED_CHARS]
        record["enriched_text"] = enriched

        embedding = embed_model.get_text_embedding(enriched)
        batch.append(build_payload(record, embedding, table))
        embedded += 1

        if len(batch) >= BATCH_SIZE:
            flushed = flush_batch(supabase, batch, table)
            failed += len(batch) - len(flushed)
            processed.update(flushed)
            batch = []

    if batch:
        flushed = flush_batch(supabase, batch, table)
        failed += len(batch) - len(flushed)

    log.info(
        "[%s] Done. embedded=%d  skipped=%d  filtered=%d  failed=%d  → %s",
        source_id, embedded, skipped, filtered, failed, table,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    from dotenv import load_dotenv
    from supabase import create_client
    load_dotenv()

    all_sources = list(SOURCE_REGISTRY)
    parser = argparse.ArgumentParser(
        description=(
            "Embed all chunked sources into their Supabase tables.\n\n"
            "Default: embeds all sources. Use --source to embed a subset.\n"
            "Source → table mapping is hardcoded in SOURCE_REGISTRY."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        nargs="+",
        metavar="SOURCE",
        choices=all_sources,
        default=all_sources,
        help="Source(s) to embed. Default: all. Choices: " + ", ".join(all_sources),
    )
    parser.add_argument(
        "--local-input",
        type=Path,
        metavar="FILE",
        default=None,
        help="Read chunks from a local JSONL file instead of S3. Requires exactly one --source.",
    )
    args = parser.parse_args()

    if args.local_input and len(args.source) > 1:
        parser.error("--local-input requires exactly one --source")

    from retrieval.embeddings import get_embedding_model
    supabase = create_client(_require_env("SUPABASE_URL"), _require_env("SUPABASE_SERVICE_KEY"))
    embed_model = get_embedding_model()
    chunked_bucket = _require_env("CHUNKED_S3_BUCKET")
    aws_region = os.getenv("AWS_REGION")

    log.info("Embedding %d source(s): %s", len(args.source), ", ".join(args.source))
    for source_id in args.source:
        embed_source(
            source_id, supabase, embed_model,
            chunked_bucket, aws_region,
            local_input=args.local_input,
        )


if __name__ == "__main__":
    main()
