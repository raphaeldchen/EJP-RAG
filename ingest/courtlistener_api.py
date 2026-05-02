import argparse
import io
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

import boto3
import pandas as pd
import requests
import tiktoken
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from core.models import Chunk

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

def get_config(local_only: bool = False) -> dict:
    raw_bucket = _require_env("RAW_S3_BUCKET")
    api_token  = _require_env("COURTLISTENER_API_TOKEN")
    cl_prefix  = os.environ.get("RAW_COURTLISTENER_S3_PREFIX", "courtlistener/").rstrip("/")
    cfg = {
        "raw_bucket":  raw_bucket,
        "raw_prefix":  f"{cl_prefix}/bulk",
        "api_token":   api_token,
    }
    if not local_only:
        cfg["chunked_bucket"] = _require_env("CHUNKED_S3_BUCKET")
        cfg["chunked_prefix"] = f"{cl_prefix}/bulk"
    return cfg

API_BASE         = "https://www.courtlistener.com/api/rest/v4/opinions"
REQUEST_TIMEOUT  = 30
RETRY_ATTEMPTS   = 3
RETRY_BACKOFF    = 2.0
RATE_LIMIT_PAUSE = 0.5
BATCH_CHECKPOINT = 500  # flush to disk every N opinions

TARGET_TOKENS    = 600
MAX_TOKENS       = 800
OVERLAP_TOKENS   = 75
ENCODING_NAME    = "cl100k_base"
MIN_CHUNK_TOKENS = 50

LOCAL_OUTPUT_DIR = Path("./data_files/chunked_output")
OUTPUT_PATH      = LOCAL_OUTPUT_DIR / "api_opinion_chunks.jsonl"
DONE_PATH        = LOCAL_OUTPUT_DIR / "api_opinion_chunks.jsonl.done"

PRECEDENTIAL_WEIGHT = {
    "Published":   1.0,
    "Unpublished": 0.5,
    "Errata":      0.1,
    "Separate":    0.4,
    "In-chambers": 0.3,
    "Relating-to": 0.2,
    "Unknown":     0.3,
    "":            0.3,
}

OPINION_TYPE_LABELS = {
    "010combined":            "combined",
    "015unamimous":           "unanimous",
    "020lead":                "majority",
    "025plurality":           "plurality",
    "030concurrence":         "concurrence",
    "035concurrence-in-part": "concurrence_in_part",
    "040dissent":             "dissent",
    "050addendum":            "addendum",
    "060remittitur":          "remittitur",
    "070rehearing":           "rehearing",
    "080on-the-merits":       "on_the_merits",
    "090on-motion":           "on_motion",
}

MAJORITY_TYPES = {"combined", "unanimous", "majority", "plurality"}

COURT_LABELS = {
    "ill":      "Illinois Supreme Court",
    "illappct": "Illinois Appellate Court",
}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_done_set() -> set[str]:
    """Return set of opinion IDs already successfully chunked."""
    if not DONE_PATH.exists():
        return set()
    with open(DONE_PATH, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def mark_done(opinion_ids: list[str]):
    """Append a batch of opinion IDs to the done file."""
    DONE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DONE_PATH, "a", encoding="utf-8") as f:
        for oid in opinion_ids:
            f.write(f"{oid}\n")


# ---------------------------------------------------------------------------
# S3 / local I/O
# ---------------------------------------------------------------------------

def s3_client():
    return boto3.client("s3")

def read_csv_from_s3(bucket: str, key: str) -> pd.DataFrame:
    log.info(f"  Reading s3://{bucket}/{key}")
    obj = s3_client().get_object(Bucket=bucket, Key=key)
    return pd.read_csv(
        io.BytesIO(obj["Body"].read()),
        low_memory=False,
        dtype=str,
        keep_default_na=False,
    )

def append_jsonl_local(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

_enc = tiktoken.get_encoding(ENCODING_NAME)

def count_tokens(text: str) -> int:
    return len(_enc.encode(text))

def token_split(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    tokens = _enc.encode(text)
    chunks = []
    start  = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(_enc.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start = end - overlap_tokens
    return chunks

def strip_html(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["p", "br", "div", "h1", "h2", "h3", "h4", "h5"]):
        tag.insert_before("\n")
    return re.sub(r"\n{3,}", "\n\n", soup.get_text()).strip()

def safe_str(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()

def safe_int(val, default: int = 0) -> int:
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default

def row_get(row, key: str) -> str:
    try:
        return safe_str(row[key])
    except (KeyError, TypeError):
        return ""


# ---------------------------------------------------------------------------
# Noise / section detection (mirrors courtlistener_chunk.py)
# ---------------------------------------------------------------------------

_NOISE_PATTERNS = [
    re.compile(r"^\s*-\s*\d+\s*-\s*$", re.MULTILINE),
    re.compile(r"\x0c"),
    re.compile(r"^\s*(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH)\s+DISTRICT\s*$",
               re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*(?:ILLINOIS\s+)?SUPREME\s+COURT\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"v\.\s{2,}[\)|\|]"),
    re.compile(r"\bNos?\.\s+\d+[-\w]+"),
    re.compile(r"^\s*NOTICE\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"Order filed|Decision filed", re.IGNORECASE),
    re.compile(r"This order was filed under|Rule 23", re.IGNORECASE),
    re.compile(r"^\s*_{5,}\s*$", re.MULTILINE),
    re.compile(r"^\s*Nos?\.\s+\d", re.MULTILINE),
]

def is_noise_chunk(text: str, token_count: int) -> bool:
    if token_count > 150:
        return False
    hits = sum(1 for p in _NOISE_PATTERNS if p.search(text))
    threshold = 1 if token_count < 60 else 2
    return hits >= threshold

_SECTION_PATTERNS = [
    re.compile(
        r"^\s*(X{0,3}(?:IX|IV|V?I{0,3}))\s*[.\-—]\s*(.{0,80})$",
        re.MULTILINE | re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(BACKGROUND|FACTS|PROCEDURAL\s+HISTORY|PROCEDURAL\s+BACKGROUND|"
        r"ANALYSIS|DISCUSSION|HOLDING|DISPOSITION|CONCLUSION|JURISDICTION|"
        r"STANDARD\s+OF\s+REVIEW|APPLICABLE\s+LAW|RELEVANT\s+STATUTES?|"
        r"PRELIMINARY\s+MATTERS?|PRIOR\s+PROCEEDINGS?|OPINION)\s*$",
        re.MULTILINE | re.IGNORECASE,
    ),
    re.compile(r"^\s*([A-Z][A-Z\s]{4,60})\s*$", re.MULTILINE),
    re.compile(
        r"^\s*([A-Z]|\d+)\s*[.\-—]\s*([A-Z][^.\n]{5,60})\s*$",
        re.MULTILINE,
    ),
]

def detect_sections(text: str) -> list[tuple[str, str]]:
    hits: list[tuple[int, str]] = []
    for pattern in _SECTION_PATTERNS:
        for m in pattern.finditer(text):
            hits.append((m.start(), m.group(0).strip()))
    if not hits:
        return [("", text)]
    hits.sort(key=lambda x: x[0])
    deduped: list[tuple[int, str]] = []
    last_pos = -100
    for pos, heading in hits:
        if pos - last_pos > 20:
            deduped.append((pos, heading))
            last_pos = pos
    sections: list[tuple[str, str]] = []
    for i, (pos, heading) in enumerate(deduped):
        start = pos + len(heading)
        end   = deduped[i + 1][0] if i + 1 < len(deduped) else len(text)
        body  = text[start:end].strip()
        if body:
            sections.append((heading, body))
    preamble = text[: deduped[0][0]].strip()
    if preamble:
        sections.insert(0, ("PREAMBLE", preamble))
    return sections if sections else [("", text)]

def _sentence_split(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z¶])", text) if s.strip()]

def _accumulate(units: list[str], separator: str) -> list[str]:
    result: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for unit in units:
        unit_tokens = count_tokens(unit)
        if unit_tokens > MAX_TOKENS:
            if current:
                result.append(separator.join(current))
                current, current_tokens = [], 0
            result.extend(token_split(unit, MAX_TOKENS, OVERLAP_TOKENS))
            continue
        if current_tokens + unit_tokens > TARGET_TOKENS and current:
            result.append(separator.join(current))
            overlap        = current[-1]
            overlap_tokens = count_tokens(overlap)
            if overlap_tokens + unit_tokens <= MAX_TOKENS:
                current        = [overlap, unit]
                current_tokens = overlap_tokens + unit_tokens
            else:
                current        = [unit]
                current_tokens = unit_tokens
        else:
            current.append(unit)
            current_tokens += unit_tokens
    if current:
        result.append(separator.join(current))
    return result

def split_section(heading: str, body: str) -> list[tuple[str, str]]:
    if count_tokens(body) <= MAX_TOKENS:
        return [(heading, body)]
    normalised = re.sub(r"(?<=[.!?]) *\n(?=[A-Z¶])", "\n\n", body)
    paragraphs = [p.strip() for p in normalised.split("\n\n") if p.strip()]
    if len(paragraphs) <= 2:
        sentences = _sentence_split(body)
        if len(sentences) > 2:
            paragraphs = sentences
    chunks = _accumulate(paragraphs, "\n\n")
    return [(heading, chunk) for chunk in chunks]


# ---------------------------------------------------------------------------
# enriched_text / display_citation helpers
# ---------------------------------------------------------------------------

def _enriched_text(
    case_name_short: str,
    date_filed: str,
    court_label: str,
    section_heading: str,
    chunk_text: str,
) -> str:
    header_parts = [x for x in [case_name_short, date_filed, court_label] if x]
    header = " | ".join(header_parts)
    if section_heading:
        return f"{header}\n{section_heading}\n\n{chunk_text}"
    return f"{header}\n\n{chunk_text}" if header else chunk_text


def _display_citation(case_name_short: str, date_filed: str) -> str:
    year = date_filed[:4] if date_filed and len(date_filed) >= 4 else ""
    if case_name_short and year:
        return f"{case_name_short} ({year})"
    return case_name_short or ""


# ---------------------------------------------------------------------------
# API fetch
# ---------------------------------------------------------------------------

def fetch_opinion(opinion_id: str, session: requests.Session) -> dict | None:
    url     = f"{API_BASE}/{opinion_id}/"
    backoff = RETRY_BACKOFF
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", backoff * attempt))
                log.warning(f"  Rate limited on {opinion_id}. Waiting {wait:.1f}s...")
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                log.warning(f"  Opinion {opinion_id} not found (404). Skipping.")
                return None
            if resp.status_code == 403:
                log.warning(f"  Opinion {opinion_id} restricted (403). Skipping.")
                return None
            if resp.status_code >= 500:
                log.warning(
                    f"  Server error {resp.status_code} for {opinion_id} "
                    f"(attempt {attempt}/{RETRY_ATTEMPTS}). Retrying in {backoff:.1f}s..."
                )
                time.sleep(backoff)
                backoff *= 2
                continue
            log.warning(f"  Status {resp.status_code} for {opinion_id}. Skipping.")
            return None
        except requests.exceptions.Timeout:
            log.warning(f"  Timeout for {opinion_id} (attempt {attempt}/{RETRY_ATTEMPTS}).")
            time.sleep(backoff)
            backoff *= 2
        except requests.exceptions.RequestException as e:
            log.warning(f"  Request error for {opinion_id}: {e}. Skipping.")
            return None
    log.error(f"  {opinion_id} failed after {RETRY_ATTEMPTS} attempts.")
    return None

def extract_text(api_response: dict) -> str:
    plain = (api_response.get("plain_text") or "").strip()
    if plain:
        return plain
    for field in ("html_with_citations", "html", "html_lawbox", "html_columbia"):
        html = (api_response.get(field) or "").strip()
        if html:
            return strip_html(html)
    return ""


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_opinion_from_api(
    opinion_id:  str,
    api_data:    dict,
    csv_row:     pd.Series,
    cluster_map: dict,
    docket_map:  dict,
) -> list[Chunk]:
    plain = extract_text(api_data)
    if not plain:
        return []
    cluster_id   = safe_str(csv_row.get("cluster_id"))
    opinion_type = safe_str(csv_row.get("type"))
    cluster     = cluster_map.get(cluster_id, {})
    docket_id   = row_get(cluster, "docket_id")
    docket      = docket_map.get(docket_id, {})
    prec_status = row_get(cluster, "precedential_status")
    court_id    = row_get(docket, "court_id")
    op_label    = OPINION_TYPE_LABELS.get(opinion_type, opinion_type)
    case_name        = row_get(cluster, "case_name")
    case_name_short  = row_get(cluster, "case_name_short")
    date_filed       = row_get(cluster, "date_filed")
    court_label      = COURT_LABELS.get(court_id, court_id)
    display_citation = _display_citation(case_name_short, date_filed)
    author           = safe_str(csv_row.get("author_str"))
    per_curiam       = safe_str(csv_row.get("per_curiam")).lower() == "true"

    sections = detect_sections(plain)
    flat: list[tuple[str, str, int]] = []
    for sec_idx, (heading, body) in enumerate(sections):
        for h, text in split_section(heading, body):
            flat.append((h, text, sec_idx))

    # Filter noise/short in one pass, carrying token_count
    prelim: list[tuple[str, str, int, int]] = []  # (heading, text, sec_idx, token_count)
    for heading, chunk_text, sec_idx in flat:
        token_count = count_tokens(chunk_text)
        if token_count < MIN_CHUNK_TOKENS:
            continue
        if is_noise_chunk(chunk_text, token_count):
            continue
        prelim.append((heading, chunk_text, sec_idx, token_count))

    chunk_total = len(prelim)
    result: list[Chunk] = []
    for chunk_index, (heading, chunk_text, sec_idx, token_count) in enumerate(prelim):
        result.append(Chunk(
            chunk_id         = f"{opinion_id}_api_c{chunk_index}",
            parent_id        = opinion_id,
            chunk_index      = chunk_index,
            chunk_total      = chunk_total,
            text             = chunk_text,
            enriched_text    = _enriched_text(case_name_short, date_filed, court_label, heading, chunk_text),
            source           = "courtlistener",
            token_count      = token_count,
            display_citation = display_citation,
            metadata={
                "fetch_method":        "api",
                "chunk_type":          "opinion_section" if heading else "opinion_paragraph",
                "section_heading":     heading,
                "section_index":       sec_idx,
                "opinion_id":          opinion_id,
                "opinion_type":        op_label,
                "is_majority":         op_label in MAJORITY_TYPES,
                "author":              author,
                "per_curiam":          per_curiam,
                "cluster_id":          cluster_id,
                "case_name":           case_name,
                "case_name_short":     case_name_short,
                "date_filed":          date_filed,
                "judges":              row_get(cluster, "judges"),
                "precedential_status": prec_status,
                "precedential_weight": PRECEDENTIAL_WEIGHT.get(prec_status, 0.3),
                "citation_count":      safe_int(row_get(cluster, "citation_count")),
                "docket_id":           docket_id,
                "court_id":            court_id,
                "court_label":         court_label,
                "docket_number":       row_get(docket, "docket_number"),
                "nature_of_suit":      row_get(docket, "nature_of_suit"),
                "cause":               row_get(docket, "cause"),
                "date_terminated":     row_get(docket, "date_terminated"),
            },
        ))
    return result


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(local_only: bool = False, limit: int = 0, ids: list[str] | None = None):
    cfg        = get_config(local_only=local_only)
    raw_bucket = cfg["raw_bucket"]
    raw_prefix = cfg["raw_prefix"]
    api_token  = cfg["api_token"]

    # Load checkpoint — skip opinions already processed
    done_set = load_done_set() if not ids else set()
    if done_set:
        log.info(f"  Resuming: {len(done_set):,} opinions already done (checkpoint).")

    log.info("Loading metadata tables from S3...")
    clusters_df = read_csv_from_s3(raw_bucket, f"{raw_prefix}/clusters.csv")
    dockets_df  = read_csv_from_s3(raw_bucket, f"{raw_prefix}/dockets.csv")
    opinions_df = read_csv_from_s3(raw_bucket, f"{raw_prefix}/opinions.csv")
    log.info(
        f"  {len(opinions_df):,} opinion IDs | "
        f"{len(clusters_df):,} clusters | {len(dockets_df):,} dockets"
    )
    cluster_map = {safe_str(r["id"]): r for _, r in clusters_df.iterrows()}
    docket_map  = {safe_str(r["id"]): r for _, r in dockets_df.iterrows()}

    opinions_df  = opinions_df.set_index("id", drop=False)
    opinion_ids  = [oid for oid in opinions_df.index if re.match(r'^\d+$', str(oid))]

    if ids:
        wanted = set(ids)
        opinion_ids = [oid for oid in opinion_ids if str(oid) in wanted]
        log.info(f"  Filtered to {len(opinion_ids)} specified IDs.")
    else:
        before = len(opinion_ids)
        opinion_ids = [oid for oid in opinion_ids if str(oid) not in done_set]
        log.info(f"  {before - len(opinion_ids):,} already done; "
                 f"{len(opinion_ids):,} remaining.")

    if limit:
        log.info(f"  Limiting to first {limit} remaining IDs.")
        opinion_ids = opinion_ids[:limit]

    total = len(opinion_ids)
    log.info(f"  Will fetch {total:,} opinions from the CourtListener API.")

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Token {api_token}",
        "User-Agent":    "illinois-legal-rag-pipeline/1.0",
    })

    all_chunks:      list[Chunk] = []
    checkpoint_buf:  list[dict]  = []
    checkpoint_ids:  list[str]   = []
    fetched          = 0
    skipped_no_text  = 0
    skipped_api_fail = 0

    log.info("Fetching opinions from API...")
    for i, opinion_id in enumerate(opinion_ids):
        api_data = fetch_opinion(opinion_id, session)
        if api_data is None:
            skipped_api_fail += 1
            # Still mark as done so reruns don't hammer the API on permanent failures
            checkpoint_ids.append(str(opinion_id))
        else:
            csv_row = opinions_df.loc[opinion_id] if opinion_id in opinions_df.index else pd.Series()
            chunks  = chunk_opinion_from_api(opinion_id, api_data, csv_row, cluster_map, docket_map)
            if not chunks:
                skipped_no_text += 1
            else:
                all_chunks.extend(chunks)
                checkpoint_buf.extend(asdict(c) for c in chunks)
                fetched += 1
            checkpoint_ids.append(str(opinion_id))

        if (i + 1) % 100 == 0:
            log.info(
                f"  {i + 1:,}/{total:,} processed | "
                f"{len(all_chunks):,} chunks | "
                f"{skipped_no_text} no-text | {skipped_api_fail} api-fail"
            )

        if len(checkpoint_buf) >= BATCH_CHECKPOINT:
            append_jsonl_local(checkpoint_buf, OUTPUT_PATH)
            mark_done(checkpoint_ids)
            checkpoint_buf.clear()
            checkpoint_ids.clear()
            log.info(f"  Checkpoint flushed → {OUTPUT_PATH}")

        time.sleep(RATE_LIMIT_PAUSE)

    # Final flush
    if checkpoint_buf or checkpoint_ids:
        if checkpoint_buf:
            append_jsonl_local(checkpoint_buf, OUTPUT_PATH)
        mark_done(checkpoint_ids)
        log.info(f"  Final flush → {OUTPUT_PATH}")

    log.info(
        f"  Done. {len(all_chunks):,} chunks from {fetched:,} opinions. "
        f"Skipped: {skipped_no_text} (no text), {skipped_api_fail} (API failure)."
    )
    if all_chunks:
        tokens = [c.token_count for c in all_chunks]
        log.info(
            f"  Chunk token stats: avg {sum(tokens)/len(tokens):.0f} | "
            f"min {min(tokens)} | max {max(tokens)}"
        )

    if local_only:
        log.info(f"  Output in: {LOCAL_OUTPUT_DIR.resolve()}")
    else:
        chunked_bucket = cfg["chunked_bucket"]
        chunked_prefix = cfg["chunked_prefix"]
        if OUTPUT_PATH.exists():
            with open(OUTPUT_PATH, "rb") as f:
                body = f.read()
            boto3.client("s3").put_object(
                Bucket=chunked_bucket,
                Key=f"{chunked_prefix}/api_opinion_chunks.jsonl",
                Body=body,
                ContentType="application/x-ndjson",
            )
            log.info(f"  Uploaded → s3://{chunked_bucket}/{chunked_prefix}/api_opinion_chunks.jsonl")
        else:
            log.warning("  No output file found — nothing uploaded to S3.")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch full opinion text from CourtListener API and chunk to JSONL."
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Write output locally without uploading to S3.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N remaining opinion IDs (0 = all). For testing.",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        help="Fetch specific opinion IDs only (bypasses done-set). e.g. --ids 11211928 11206006",
    )
    args = parser.parse_args()
    run(local_only=args.local_only, limit=args.limit, ids=args.ids)


if __name__ == "__main__":
    main()
