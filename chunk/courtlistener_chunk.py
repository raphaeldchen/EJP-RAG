import argparse
import io
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import boto3
import pandas as pd
import tiktoken
from bs4 import BeautifulSoup
from dotenv import load_dotenv

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
    raw_bucket     = _require_env("RAW_S3_BUCKET")
    chunked_bucket = _require_env("CHUNKED_S3_BUCKET")
    cl_prefix      = os.environ.get("RAW_COURTLISTENER_S3_PREFIX", "courtlistener/").rstrip("/")
    return {
        "raw_bucket":     raw_bucket,
        "raw_prefix":     f"{cl_prefix}/bulk",
        "chunked_bucket": chunked_bucket,
        "chunked_prefix": f"{cl_prefix}/bulk",
    }

TARGET_TOKENS    = 600   # soft target chunk size
MAX_TOKENS       = 800   # hard ceiling; chunks must not exceed this
OVERLAP_TOKENS   = 75    # token overlap carried between adjacent chunks (when safe)
ENCODING_NAME    = "cl100k_base"
MIN_CHUNK_TOKENS = 50    # discard fragments shorter than this

LOCAL_OUTPUT_DIR = Path("./data_files/chunked_output")

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

@dataclass
class OpinionChunk:
    chunk_id:            str
    chunk_index:         int
    chunk_type:          str    # "opinion_section" | "opinion_paragraph"
    source:              str    # always "courtlistener"
    text:                str
    token_count:         int
    section_heading:     str
    section_index:       int
    opinion_id:          str
    opinion_type:        str
    is_majority:         bool
    author:              str
    per_curiam:          bool
    cluster_id:          str
    case_name:           str
    case_name_short:     str
    date_filed:          str
    judges:              str
    precedential_status: str
    precedential_weight: float
    citation_count:      int
    docket_id:           str
    court_id:            str
    court_label:         str
    docket_number:       str
    nature_of_suit:      str
    cause:               str
    date_terminated:     str


@dataclass
class ParentheticalChunk:
    chunk_id:               str
    chunk_type:             str    # always "parenthetical"
    source:                 str    # always "courtlistener"
    text:                   str
    token_count:            int
    describing_opinion_id:  str
    described_opinion_id:   str
    score:                  float
    case_name:              str
    date_filed:             str
    court_id:               str
    court_label:            str
    precedential_status:    str
    precedential_weight:    float

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

def write_jsonl_to_s3(records: list[dict], bucket: str, key: str):
    log.info(f"  Writing {len(records):,} records → s3://{bucket}/{key}")
    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
    s3_client().put_object(
        Bucket=bucket,
        Key=key,
        Body=body.encode("utf-8"),
        ContentType="application/x-ndjson",
    )
    log.info("  Upload complete.")

def write_jsonl_local(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    log.info(f"  Saved locally: {path}  ({len(records):,} records)")

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

def safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

def row_get(row, key: str) -> str:
    try:
        return safe_str(row[key])
    except (KeyError, TypeError):
        return ""

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
        r"^\s*(X{0,3}(?:IX|IV|V?I{0,3}))\s*[.\-\u2014]\s*(.{0,80})$",
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
        r"^\s*([A-Z]|\d+)\s*[.\-\u2014]\s*([A-Z][^.\n]{5,60})\s*$",
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
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\u00b6])", text) if s.strip()]

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
    normalised = re.sub(r"(?<=[.!?]) *\n(?=[A-Z\u00b6])", "\n\n", body)
    paragraphs = [p.strip() for p in normalised.split("\n\n") if p.strip()]
    if len(paragraphs) <= 2:
        sentences = _sentence_split(body)
        if len(sentences) > 2:
            paragraphs = sentences
    chunks = _accumulate(paragraphs, "\n\n")
    return [(heading, chunk) for chunk in chunks]

def build_lookup_maps(
    clusters_df: pd.DataFrame,
    dockets_df: pd.DataFrame,
) -> tuple[dict, dict]:
    cluster_map = {safe_str(row["id"]): row for _, row in clusters_df.iterrows()}
    docket_map  = {safe_str(row["id"]): row for _, row in dockets_df.iterrows()}
    return cluster_map, docket_map

def chunk_opinion(
    opinion_row: pd.Series,
    cluster_map: dict,
    docket_map: dict,
) -> list[OpinionChunk]:
    opinion_id   = safe_str(opinion_row.get("id"))
    cluster_id   = safe_str(opinion_row.get("cluster_id"))
    opinion_type = safe_str(opinion_row.get("type"))
    plain = safe_str(opinion_row.get("plain_text"))
    if not plain:
        html  = safe_str(opinion_row.get("html_with_citations"))
        plain = strip_html(html) if html else ""
    if not plain:
        return []
    cluster     = cluster_map.get(cluster_id, {})
    docket_id   = row_get(cluster, "docket_id")
    docket      = docket_map.get(docket_id, {})
    prec_status = row_get(cluster, "precedential_status")
    court_id    = row_get(docket, "court_id")
    op_label    = OPINION_TYPE_LABELS.get(opinion_type, opinion_type)
    sections = detect_sections(plain)
    flat: list[tuple[str, str, int]] = []
    for sec_idx, (heading, body) in enumerate(sections):
        for h, text in split_section(heading, body):
            flat.append((h, text, sec_idx))
    result: list[OpinionChunk] = []
    for chunk_idx, (heading, chunk_text, sec_idx) in enumerate(flat):
        token_count = count_tokens(chunk_text)
        if token_count < MIN_CHUNK_TOKENS:
            continue
        if is_noise_chunk(chunk_text, token_count):
            continue
        result.append(OpinionChunk(
            chunk_id            = f"{opinion_id}_{chunk_idx}",
            chunk_index         = chunk_idx,
            chunk_type          = "opinion_section" if heading else "opinion_paragraph",
            source              = "courtlistener",
            text                = chunk_text,
            token_count         = token_count,
            section_heading     = heading,
            section_index       = sec_idx,
            opinion_id          = opinion_id,
            opinion_type        = op_label,
            is_majority         = op_label in MAJORITY_TYPES,
            author              = safe_str(opinion_row.get("author_str")),
            per_curiam          = safe_str(opinion_row.get("per_curiam")).lower() == "true",
            cluster_id          = cluster_id,
            case_name           = row_get(cluster, "case_name"),
            case_name_short     = row_get(cluster, "case_name_short"),
            date_filed          = row_get(cluster, "date_filed"),
            judges              = row_get(cluster, "judges"),
            precedential_status = prec_status,
            precedential_weight = PRECEDENTIAL_WEIGHT.get(prec_status, 0.3),
            citation_count      = safe_int(row_get(cluster, "citation_count")),
            docket_id           = docket_id,
            court_id            = court_id,
            court_label         = COURT_LABELS.get(court_id, court_id),
            docket_number       = row_get(docket, "docket_number"),
            nature_of_suit      = row_get(docket, "nature_of_suit"),
            cause               = row_get(docket, "cause"),
            date_terminated     = row_get(docket, "date_terminated"),
        ))
    return result

def chunk_parentheticals(
    parentheticals_df: pd.DataFrame,
    cluster_map: dict,
    docket_map: dict,
    opinion_to_cluster: dict[str, str],
) -> list[ParentheticalChunk]:
    result: list[ParentheticalChunk] = []
    for _, row in parentheticals_df.iterrows():
        text = safe_str(row.get("text"))
        if not text:
            continue
        describing_id = safe_str(row.get("describing_opinion_id"))
        cluster_id    = opinion_to_cluster.get(describing_id, "")
        cluster       = cluster_map.get(cluster_id, {})
        docket_id     = row_get(cluster, "docket_id")
        docket        = docket_map.get(docket_id, {})
        court_id      = row_get(docket, "court_id")
        prec_status   = row_get(cluster, "precedential_status")
        result.append(ParentheticalChunk(
            chunk_id              = f"par_{safe_str(row.get('id'))}",
            chunk_type            = "parenthetical",
            source                = "courtlistener",
            text                  = text,
            token_count           = count_tokens(text),
            describing_opinion_id = describing_id,
            described_opinion_id  = safe_str(row.get("described_opinion_id")),
            score                 = safe_float(row.get("score")),
            case_name             = row_get(cluster, "case_name"),
            date_filed            = row_get(cluster, "date_filed"),
            court_id              = court_id,
            court_label           = COURT_LABELS.get(court_id, court_id),
            precedential_status   = prec_status,
            precedential_weight   = PRECEDENTIAL_WEIGHT.get(prec_status, 0.3),
        ))
    return result

def run(local_only: bool = False, limit: int = 0):
    cfg            = get_config()
    raw_bucket     = cfg["raw_bucket"]
    raw_prefix     = cfg["raw_prefix"]
    chunked_bucket = cfg["chunked_bucket"]
    chunked_prefix = cfg["chunked_prefix"]
    log.info("Loading raw tables from S3...")
    clusters_df       = read_csv_from_s3(raw_bucket, f"{raw_prefix}/clusters.csv")
    dockets_df        = read_csv_from_s3(raw_bucket, f"{raw_prefix}/dockets.csv")
    opinions_df       = read_csv_from_s3(raw_bucket, f"{raw_prefix}/opinions.csv")
    parentheticals_df = read_csv_from_s3(raw_bucket, f"{raw_prefix}/parentheticals.csv")
    log.info(
        f"  {len(opinions_df):,} opinions | {len(clusters_df):,} clusters | "
        f"{len(dockets_df):,} dockets | {len(parentheticals_df):,} parentheticals"
    )
    has_html  = (opinions_df["html_with_citations"].str.strip() != "").sum()
    has_plain = (opinions_df["plain_text"].str.strip() != "").sum() \
                if "plain_text" in opinions_df.columns else 0
    log.info(
        f"  Text coverage: {has_html:,} html_with_citations | "
        f"{has_plain:,} plain_text | "
        f"{len(opinions_df) - max(has_html, has_plain):,} have NO text"
    )
    log.info("Building metadata lookup maps...")
    cluster_map, docket_map = build_lookup_maps(clusters_df, dockets_df)
    opinion_to_cluster = {
        safe_str(row["id"]): safe_str(row["cluster_id"])
        for _, row in opinions_df.iterrows()
    }
    opinions_df = opinions_df.assign(
        _text_len=(
            opinions_df["html_with_citations"].str.len().fillna(0)
            + opinions_df["plain_text"].str.len().fillna(0)
        )
    ).sort_values("_text_len", ascending=False).drop(columns="_text_len")
    if limit:
        log.info(f"  Limiting to first {limit} opinions (by text length).")
        opinions_df = opinions_df.head(limit)
    log.info("Chunking opinions...")
    opinion_chunks: list[OpinionChunk] = []
    opinions_with_chunks = 0
    skipped_no_text   = 0
    skipped_too_short = 0
    for i, (_, row) in enumerate(opinions_df.iterrows()):
        chunks = chunk_opinion(row, cluster_map, docket_map)
        if not chunks:
            skipped_no_text += 1
            continue
        useful = [c for c in chunks if c.token_count >= MIN_CHUNK_TOKENS]
        if useful:
            opinion_chunks.extend(useful)
            opinions_with_chunks += 1
        else:
            skipped_too_short += 1
        if (i + 1) % 1000 == 0:
            log.info(f"  {i + 1:,} opinions processed → {len(opinion_chunks):,} chunks...")
    log.info(
        f"  Done. {len(opinion_chunks):,} chunks produced from {opinions_with_chunks:,} opinions. "
        f"Skipped: {skipped_no_text} (no text), {skipped_too_short} (too short <{MIN_CHUNK_TOKENS} tokens)."
    )
    log.info("Chunking parentheticals...")
    par_chunks = chunk_parentheticals(
        parentheticals_df, cluster_map, docket_map, opinion_to_cluster
    )
    log.info(f"  {len(par_chunks):,} parenthetical chunks produced.")
    opinion_records = [asdict(c) for c in opinion_chunks]
    par_records     = [asdict(c) for c in par_chunks]
    if opinion_records:
        tokens = [r["token_count"] for r in opinion_records]
        log.info(
            f"  Chunk token stats: avg {sum(tokens)/len(tokens):.0f} | "
            f"min {min(tokens)} | max {max(tokens)}"
        )
    if local_only:
        write_jsonl_local(opinion_records, LOCAL_OUTPUT_DIR / "opinion_chunks.jsonl")
        write_jsonl_local(par_records,     LOCAL_OUTPUT_DIR / "parenthetical_chunks.jsonl")
        log.info(f"  Output in: {LOCAL_OUTPUT_DIR.resolve()}")
    else:
        write_jsonl_to_s3(opinion_records, chunked_bucket, f"{chunked_prefix}/opinion_chunks.jsonl")
        write_jsonl_to_s3(par_records,     chunked_bucket, f"{chunked_prefix}/parenthetical_chunks.jsonl")
        log.info(f"  Output at s3://{chunked_bucket}/{chunked_prefix}/")

def main():
    parser = argparse.ArgumentParser(
        description="Chunk Illinois legal corpus CSV → JSONL (S3)"
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
        help="Process only the first N opinions by text length (0 = all). For testing.",
    )
    args = parser.parse_args()
    run(local_only=args.local_only, limit=args.limit)

if __name__ == "__main__":
    main()