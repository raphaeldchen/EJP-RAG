import argparse
import dataclasses
import io
import json
import logging
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import boto3
import pdfplumber
import tiktoken
from dotenv import load_dotenv
from core.models import Chunk

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Separate logger for skipped chunks — only active when --debug is passed
skip_log = logging.getLogger("skipped_chunks")
skip_log.propagate = False  # Don't bubble up to root logger

def enable_debug_logging(log_path: str = "skipped_chunks.log") -> None:
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    skip_log.addHandler(handler)
    skip_log.setLevel(logging.DEBUG)
    log.info(f"Debug mode enabled — skipped chunks will be written to '{log_path}'")

# Much lower threshold - legal text often has short but important content
MIN_CHUNK_TOKENS = 10  # About 40 characters minimum
# Patterns for content that should be kept even if very short
IMPORTANT_SHORT_PATTERNS = [
    r"\([a-z0-9]\)",           # Subsection markers like (a), (1)
    r"Rule\s+\d+",              # Rule references
    r"§\s+\d+",                 # Section symbols
    r"(?:shall|must|may)\s+",   # Obligation language
    r"defined\s+as",            # Definitions
    r"means\s+",                # Definitions
    r"includes?\s+",             # Definitions
]
# Patterns for content that can be safely skipped even if longer
SKIP_PATTERNS = [
    r"^Rule\s+\d+\.?\s+Reserved\.?$",  # Reserved rules
    r"^\[\s*PAGE\s+\d+\s*\]$",          # Page markers
    r"^Amended\s+.*$",                   # Amendment lines alone
    r"^Effective\s+.*$",                  # Effective date lines alone
    r"^Adopted\s+.*$",                    # Adoption lines alone
    r"^Committee Comments?$",              # Committee Comment headers (will be handled with content)
]

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def should_keep_text(text: str) -> bool:
    text_stripped = text.strip()
    # Skip obvious metadata
    for pattern in SKIP_PATTERNS:
        if re.match(pattern, text_stripped, re.IGNORECASE):
            return False
    # Keep if it matches important patterns
    for pattern in IMPORTANT_SHORT_PATTERNS:
        if re.search(pattern, text_stripped, re.IGNORECASE):
            return True
    # Keep if it looks like a definition or element (starts with number/letter and has content)
    if re.match(r'^\([a-z0-9]\)\s+\S', text_stripped):
        return True
    # Keep if it's a header that introduces content
    if re.match(r'^[A-Z][a-z]+\.?\s+\S', text_stripped) and len(text_stripped) < 100:
        return True
    # Default: keep if it has any alphanumeric content
    return bool(re.search(r'\w', text_stripped))

def get_config() -> dict:
    required = {
        "raw_bucket":      "RAW_S3_BUCKET",
        "raw_prefix":      "SUPREME_COURT_RULES_S3_PREFIX",
        "chunked_bucket":  "CHUNKED_S3_BUCKET",
        "chunked_prefix":  "SUPREME_COURT_RULES_S3_PREFIX",
    }
    config = {}
    missing = []
    for key, env_var in required.items():
        value = os.environ.get(env_var)
        if not value:
            missing.append(env_var)
        else:
            # Normalise: strip trailing slashes, we'll add them explicitly
            config[key] = value.rstrip("/")
    if missing:
        log.error(f"Missing required environment variables: {', '.join(missing)}")
        sys.exit(1)
    return config

def list_pdfs(s3: "boto3.client", bucket: str, prefix: str) -> list[str]:
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/"):
        for obj in page.get("Contents", []):
            if obj["Key"].lower().endswith(".pdf"):
                keys.append(obj["Key"])
    return keys

def download_pdf_bytes(s3: "boto3.client", bucket: str, key: str) -> bytes:
    log.info(f"Downloading s3://{bucket}/{key}")
    response = s3.get_object(Bucket=bucket, Key=key)
    return response["Body"].read()

def upload_jsonl(s3: "boto3.client", bucket: str, key: str, chunks: list[dict]) -> None:
    lines = "\n".join(json.dumps(chunk, ensure_ascii=False) for chunk in chunks) + "\n"
    body = lines.encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/x-ndjson")
    log.info(f"Uploaded {len(chunks)} chunks → s3://{bucket}/{key}")

def extract_pages(pdf_bytes: bytes) -> list[dict]:
    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(x_tolerance=2, y_tolerance=2)
            if text:
                pages.append({"page_num": i, "text": text.strip()})
    return pages

def merge_pages_to_text(pages: list[dict]) -> str:
    return "\n\n".join(f"[PAGE {p['page_num']}]\n{p['text']}" for p in pages)

# Main article headers: "Article I. General Rules"
ARTICLE_HEADER_RE = re.compile(
    r"^(?:ARTICLE|Article)\s+([IVX]+)\.\s+(.+)$",
    re.MULTILINE
)
# Part headers: "Part A. Process and Notice" or "Part E. Discovery"
PART_HEADER_RE = re.compile(
    r"^(?:PART|Part)\s+([A-Z])\.\s+(.+)$",
    re.MULTILINE
)
# Rule headers: "Rule 1. Applicability" or "Rule 101. Summons and Original Process"
RULE_HEADER_RE = re.compile(
    r"^(?:RULE|Rule)\s+(\d{1,4}[A-Z]?(?:\.[A-Z])?)\b\.?\s+(.+?)(?:\s+\([a-z]\))?$",
    re.MULTILINE
)
# Subsection markers like "(a)" or "(1)" for splitting long rules
SUBSECTION_RE = re.compile(
    r"^\(([a-z]|[0-9]+)\)\s+",
    re.MULTILINE
)
# Split rules only at letter-level subsections — numeric items (1)(2)(3) stay
# with their parent letter subsection to preserve introductory context.
LETTER_SUBSECTION_RE = re.compile(
    r"^\(([a-z])\)\s+",
    re.MULTILINE
)
# Committee Comments section detection
COMMITTEE_COMMENT_RE = re.compile(
    r"(?:Committee Comments?|COMMITTEE COMMENTS?|Comment|COMMENT)\s*.*?(?=\n\n|\Z)",
    re.DOTALL | re.IGNORECASE
)
# Amendment history lines
AMENDMENT_RE = re.compile(
    r"(?:Amended|Adopted|Effective)\s+[A-Za-z]+\s+\d{1,2},\s+\d{4}.*?(?=\n\n|\Z)",
    re.DOTALL
)
# Cross-reference pattern
CROSS_REF_RE = re.compile(
    r"\b(?:Rule[s]?|Section|§)\s+(\d{1,4}[A-Z]?(?:\s*(?:,|and|&)\s*\d{1,4}[A-Z]?)*)",
    re.IGNORECASE
)

class DocumentHierarchy:
    def __init__(self):
        self.current_article_number: Optional[str] = None
        self.current_article_title: Optional[str] = None
        self.current_part_letter: Optional[str] = None
        self.current_part_title: Optional[str] = None
        self.current_rule_number: Optional[str] = None
        self.current_rule_title: Optional[str] = None
        self.effective_date: Optional[str] = None
    
    def update_from_line(self, line: str) -> Tuple[str, Optional[Dict]]:
        # Check for article
        article_match = ARTICLE_HEADER_RE.match(line)
        if article_match:
            self.current_article_number = article_match.group(1)
            self.current_article_title = article_match.group(2).strip()
            self.current_part_letter = None
            self.current_part_title = None
            return 'article', {
                'article_number': self.current_article_number,
                'article_title': self.current_article_title
            }
        # Check for part
        part_match = PART_HEADER_RE.match(line)
        if part_match:
            self.current_part_letter = part_match.group(1)
            self.current_part_title = part_match.group(2).strip()
            return 'part', {
                'part_letter': self.current_part_letter,
                'part_title': self.current_part_title
            }
        # Check for rule
        rule_match = RULE_HEADER_RE.match(line)
        if rule_match:
            self.current_rule_number = rule_match.group(1)
            self.current_rule_title = rule_match.group(2).strip()
            return 'rule', {
                'rule_number': self.current_rule_number,
                'rule_title': self.current_rule_title
            }
        # Check for effective date in amendment text
        if "effective" in line.lower() and re.search(r"\b\d{4}\b", line):
            date_match = re.search(r"(?:effective|eff\.)\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})", line, re.IGNORECASE)
            if date_match:
                self.effective_date = date_match.group(1)   
        return 'text', None

def estimate_tokens(text: str) -> int:
    return len(text) // 4

def extract_cross_references(text: str) -> list[str]:
    refs = []
    for match in CROSS_REF_RE.finditer(text):
        numbers = re.findall(r"\d{1,4}[A-Z]?(?:\.[A-Z])?", match.group(1))
        refs.extend(numbers)
    return sorted(set(refs))

def extract_amendment_history(text: str) -> Optional[str]:
    match = AMENDMENT_RE.search(text)
    return match.group(0).strip() if match else None

def should_split_rule(text: str) -> bool:
    # Split if rule has multiple subsections and is long (>1000 chars)
    subsection_matches = list(SUBSECTION_RE.finditer(text))
    return len(subsection_matches) >= 3 and len(text) > 1000

def split_rule_into_subsections(rule_text: str, rule_number: str) -> List[Tuple[str, str, str]]:
    """Split a rule at letter-subsection boundaries only.

    Numeric items like (1)(2)(3) stay with their parent letter subsection.
    Intro text (before the first letter subsection) is prepended to the first
    letter subsection so it is never emitted as a standalone orphan chunk.
    """
    sections = []
    lines = rule_text.split('\n')
    current_section: List[str] = []
    current_subsection: Optional[str] = None
    current_title: Optional[str] = None
    carry_forward: List[str] = []  # intro lines to prepend to the first subsection

    for line in lines:
        subsection_match = LETTER_SUBSECTION_RE.match(line.strip())
        if subsection_match and current_section:
            saved_text = '\n'.join(current_section).strip()
            if current_subsection is None:
                # This is intro text — carry forward rather than emit standalone
                carry_forward = current_section[:]
            else:
                prefix = '\n'.join(carry_forward).strip()
                full_text = (prefix + '\n' + saved_text).strip() if prefix else saved_text
                sections.append((current_subsection, current_title or f"Rule {rule_number}", full_text))
                carry_forward = []
            current_section = []
        if subsection_match:
            current_subsection = subsection_match.group(1)
            title_part = line.strip()[len(subsection_match.group(0)):].strip()
            current_title = f"Rule {rule_number}({current_subsection})"
            if title_part and len(title_part) < 100:
                current_title += f" — {title_part}"
            current_section.append(line)
        else:
            current_section.append(line)

    # Emit the final section
    if current_section:
        saved_text = '\n'.join(current_section).strip()
        prefix = '\n'.join(carry_forward).strip()
        full_text = (prefix + '\n' + saved_text).strip() if prefix else saved_text
        sections.append((
            current_subsection or "full",
            f"Rule {rule_number}" + (f"({current_subsection})" if current_subsection else ""),
            full_text,
        ))

    return sections

def _build_chunk(
    text: str,
    source_key: str,
    hierarchy: DocumentHierarchy,
    content_type: str = "rule_text",
    subsection_id: Optional[str] = None,
    committee_comments: Optional[str] = None,
) -> Chunk:
    # Build hierarchical path
    hierarchical_path_parts = []
    if hierarchy.current_article_number:
        hierarchical_path_parts.append(f"Article {hierarchy.current_article_number}")
    if hierarchy.current_part_letter:
        hierarchical_path_parts.append(f"Part {hierarchy.current_part_letter}")
    if hierarchy.current_rule_number:
        rule_ref = f"Rule {hierarchy.current_rule_number}"
        if subsection_id:
            rule_ref += f"({subsection_id})"
        hierarchical_path_parts.append(rule_ref)
    hierarchical_path = " → ".join(hierarchical_path_parts) if hierarchical_path_parts else ""

    # Build rule title string
    rule_number = hierarchy.current_rule_number or ""
    rule_title_parts = []
    if rule_number:
        rule_title_parts.append(f"Rule {rule_number}")
        if subsection_id:
            rule_title_parts.append(f"({subsection_id})")
        if hierarchy.current_rule_title and not subsection_id:
            rule_title_parts.append(f": {hierarchy.current_rule_title}")
    rule_title = " ".join(rule_title_parts)

    # display_citation: "Rule N: Title" for titled rules, "Rule N" for untitled
    if rule_number and hierarchy.current_rule_title and not subsection_id:
        display_citation = f"Rule {rule_number} — {hierarchy.current_rule_title}"
    elif rule_number:
        display_citation = f"Rule {rule_number}"
        if subsection_id:
            display_citation += f"({subsection_id})"
    else:
        display_citation = hierarchical_path

    # enriched_text: hierarchical context + text
    context_parts = []
    if hierarchy.current_article_title:
        context_parts.append(f"[{hierarchy.current_article_title}]")
    if hierarchy.current_part_title:
        context_parts.append(hierarchy.current_part_title)
    if rule_title:
        context_parts.append(rule_title)
    enriched = "\n".join(context_parts) + "\n\n" + text if context_parts else text

    return Chunk(
        chunk_id=str(uuid.uuid4()),
        parent_id=source_key,
        chunk_index=0,          # assigned later in chunk_document
        chunk_total=1,          # assigned later in chunk_document
        text=text,
        enriched_text=enriched,
        source="illinois_supreme_court_rules",
        token_count=count_tokens(text),
        display_citation=display_citation,
        metadata={
            "source_s3_key":      source_key,
            "content_type":       content_type,
            "hierarchical_path":  hierarchical_path,
            "article_number":     hierarchy.current_article_number,
            "article_title":      hierarchy.current_article_title,
            "part_letter":        hierarchy.current_part_letter,
            "part_title":         hierarchy.current_part_title,
            "rule_number":        rule_number,
            "rule_title":         rule_title,
            "subsection_id":      subsection_id,
            "effective_date":     hierarchy.effective_date,
            "amendment_history":  extract_amendment_history(text),
            "committee_comments": committee_comments,
            "cross_references":   extract_cross_references(text),
        },
    )

def chunk_document(full_text: str, source_key: str) -> list[Chunk]:
    chunks = []
    hierarchy = DocumentHierarchy()
    # Strip [PAGE N] markers injected by merge_pages_to_text before line processing.
    full_text = re.sub(r"^\[PAGE \d+\]\n?", "", full_text, flags=re.MULTILINE)
    # Split document into lines for processing
    lines = full_text.split('\n')
    current_rule_lines = []
    in_rule = False
    in_committee_comment = False
    committee_comment_lines = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        # Update hierarchy based on this line
        line_type, metadata = hierarchy.update_from_line(line)
        # Handle article/part headers as their own chunks for the hierarchy
        if line_type in ('article', 'part'):
            # Save any pending rule
            if current_rule_lines:
                rule_text = '\n'.join(current_rule_lines).strip()
                if rule_text and should_keep_text(rule_text):
                    chunks.extend(_process_rule_text(rule_text, source_key, hierarchy))
                current_rule_lines = []
                in_rule = False
            # Create a header chunk
            chunks.append(_build_chunk(
                text=line,
                source_key=source_key,
                hierarchy=hierarchy,
                content_type=f"{line_type}_header"
            ))
            i += 1
            continue
        # Check for rule start
        if line_type == 'rule':
            # Save previous rule if any
            if current_rule_lines:
                rule_text = '\n'.join(current_rule_lines).strip()
                if rule_text and should_keep_text(rule_text):
                    chunks.extend(_process_rule_text(rule_text, source_key, hierarchy))
            # Start new rule
            current_rule_lines = [line]
            in_rule = True
            in_committee_comment = False
            committee_comment_lines = []
            i += 1
            continue
        # Check for committee comments (they often follow rules)
        if in_rule and COMMITTEE_COMMENT_RE.match(line):
            if committee_comment_lines:
                # Save previous committee comment
                comment_text = '\n'.join(committee_comment_lines).strip()
                if comment_text and should_keep_text(comment_text):
                    chunks.append(_build_chunk(
                        text=comment_text,
                        source_key=source_key,
                        hierarchy=hierarchy,
                        content_type="committee_comment"
                    ))
                committee_comment_lines = []
            in_committee_comment = True
            committee_comment_lines.append(line)
            i += 1
            continue
        # Add line to appropriate buffer
        if in_committee_comment:
            committee_comment_lines.append(line)
        elif in_rule:
            current_rule_lines.append(line)
        i += 1
    # Handle final rule
    if current_rule_lines:
        rule_text = '\n'.join(current_rule_lines).strip()
        if rule_text and should_keep_text(rule_text):
            chunks.extend(_process_rule_text(rule_text, source_key, hierarchy))
    # Handle final committee comment
    if committee_comment_lines:
        comment_text = '\n'.join(committee_comment_lines).strip()
        if comment_text and should_keep_text(comment_text):
            chunks.append(_build_chunk(
                text=comment_text,
                source_key=source_key,
                hierarchy=hierarchy,
                content_type="committee_comment"
            ))
    for i, chunk in enumerate(chunks):
        chunk.chunk_index = i
        chunk.chunk_total = len(chunks)
    return chunks

def _process_rule_text(rule_text: str, source_key: str, hierarchy: DocumentHierarchy) -> list[Chunk]:
    # Check if we should split this rule
    if should_split_rule(rule_text):
        sections = split_rule_into_subsections(rule_text, hierarchy.current_rule_number or "unknown")
        
        chunks = []
        for subsection_id, title, text in sections:
            if should_keep_text(text):
                chunks.append(_build_chunk(
                    text=text,
                    source_key=source_key,
                    hierarchy=hierarchy,
                    content_type="rule_subsection",
                    subsection_id=subsection_id
                ))
            elif skip_log.handlers:
                skip_log.debug(
                    f"\n{'='*60}\n"
                    f"SKIPPED subsection {hierarchy.current_rule_number}({subsection_id}) "
                    f"| {estimate_tokens(text)} tokens\n"
                    f"--- Text ---\n{text[:500]}...\n"
                    f"{'='*60}"
                )
        return chunks
    else:
        # Single chunk for the whole rule
        if should_keep_text(rule_text):
            return [_build_chunk(
                text=rule_text,
                source_key=source_key,
                hierarchy=hierarchy,
                content_type="rule_text"
            )]
        else:
            if skip_log.handlers:
                skip_log.debug(
                    f"\n{'='*60}\n"
                    f"SKIPPED rule {hierarchy.current_rule_number} | {estimate_tokens(rule_text)} tokens\n"
                    f"--- Text ---\n{rule_text[:500]}...\n"
                    f"{'='*60}"
                )
            return []

def process_pdf(s3: "boto3.client", pdf_key: str, config: dict) -> int:
    pdf_bytes = download_pdf_bytes(s3, config["raw_bucket"], pdf_key)
    pages = extract_pages(pdf_bytes)
    if not pages:
        log.warning(f"No text extracted from {pdf_key}. Possibly scanned — consider OCR preprocessing.")
        return 0
    full_text = merge_pages_to_text(pages)
    chunks = chunk_document(full_text, source_key=pdf_key)
    pdf_filename = Path(pdf_key).stem
    output_key = f"{config['chunked_prefix']}/{pdf_filename}_chunks.jsonl"
    serialized = [dataclasses.asdict(c) for c in chunks]
    upload_jsonl(s3, config["chunked_bucket"], output_key, serialized)
    log.info(f"Processed {pdf_key}: {len(chunks)} chunks created")
    return len(chunks)

def main():
    parser = argparse.ArgumentParser(description="Chunk Illinois Supreme Court Rules PDFs to JSONL on S3")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Dump skipped chunks (text + position) to skipped_chunks.log for inspection",
    )
    parser.add_argument(
        "--debug-log",
        type=str,
        default="skipped_chunks.log",
        metavar="PATH",
        help="Path for the debug log file (default: skipped_chunks.log)",
    )
    args = parser.parse_args()
    if args.debug:
        enable_debug_logging(args.debug_log)
    config = get_config()
    s3 = boto3.client("s3")
    log.info(f"Listing PDFs in s3://{config['raw_bucket']}/{config['raw_prefix']}/")
    pdf_keys = list_pdfs(s3, config["raw_bucket"], config["raw_prefix"])
    if not pdf_keys:
        log.error(f"No PDFs found under s3://{config['raw_bucket']}/{config['raw_prefix']}/")
        sys.exit(1)
    log.info(f"Found {len(pdf_keys)} PDF(s)")
    total_chunks = 0
    for pdf_key in sorted(pdf_keys):
        total_chunks += process_pdf(s3, pdf_key, config)
    log.info(f"Done. Total chunks produced: {total_chunks}")
    if args.debug:
        log.info(f"Skipped chunk details written to '{args.debug_log}'")

if __name__ == "__main__":
    main()