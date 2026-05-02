"""Shared utilities for opinion chunking (CAP and CourtListener)."""

import math
import re

import tiktoken
from bs4 import BeautifulSoup

ENCODING_NAME    = "cl100k_base"
TARGET_TOKENS    = 600
MAX_TOKENS       = 800
OVERLAP_TOKENS   = 75
MIN_CHUNK_TOKENS = 50

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
    if val is None or (isinstance(val, float) and math.isnan(val)):
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
        r"^\s*(X{0,3}(?:IX|IV|VI{0,3}|V|I{1,3})|X{1,3})\s*[.\-—]\s*(.{0,80})$",
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


def _opinion_enriched_text(
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


def _opinion_display_citation(case_name_short: str, date_filed: str) -> str:
    year = date_filed[:4] if date_filed and len(date_filed) >= 4 else ""
    if case_name_short and year:
        return f"{case_name_short} ({year})"
    return case_name_short or ""
