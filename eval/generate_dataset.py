"""
Generates a golden retrieval evaluation dataset.

Process:
  1. Calls Claude to generate realistic test queries (query-first) — no citations shown,
     so query diversity is unconstrained by what happens to be in the DB
  2. Fetches all distinct citations from Supabase (paginated)
  3. Calls Claude to annotate each query with the citations a correct retrieval system
     must return, choosing only from the verified DB citations
  4. Validates every expected_citation against Supabase — strips invalid citations, drops
     cases only when no valid citations remain
  5. Saves validated cases to data_files/eval_files/dataset.json

Usage:
    python -m eval.generate_dataset
    python -m eval.generate_dataset --n 80 --output path/to/dataset.json
"""

import json
import argparse
import sys
import time
from pathlib import Path
from collections import defaultdict

import anthropic
from supabase import create_client

sys.path.insert(0, str(Path(__file__).parent.parent))
from retrieval.config import (
    SUPABASE_URL,
    SUPABASE_SERVICE_KEY,
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
)

DATASET_PATH = Path(__file__).parent.parent / "data_files" / "eval_files" / "dataset.json"

_MAX_ISCR_RULES = 40
_FALLBACK_PER_CHAPTER = 5   # citations from chapters NOT detected as relevant to the batch
_QUERY_BATCH_SIZE = 30      # queries generated per API call
_ANNOTATION_BATCH_SIZE = 20 # queries annotated per API call
_INTER_CALL_DELAY = 65      # seconds between API calls — stays under 10K input tokens/min limit

# Maps ILCS chapter prefix → keywords that signal a query is about that chapter.
# When any keyword matches in the batch's query text, that chapter's full citation
# list is included in the annotation prompt instead of the small fallback sample.
_CHAPTER_KEYWORDS: dict[str, list[str]] = {
    "720": [
        "offense", "crime", "criminal", "assault", "battery", "theft", "murder",
        "robbery", "burglary", "dui", "drug", "cannabis", "firearm", "weapon",
        "sexual", "arson", "fraud", "stalking", "domestic", "homicide", "attempt",
        "intimidation", "kidnap", "trespass", "disorderly",
    ],
    "725": [
        "procedure", "arrest", "trial", "evidence", "bail", "speedy", "discovery",
        "jury", "plea", "indictment", "subpoena", "search", "warrant",
        "motion", "suppress", "hearing", "preliminary", "prosecution", "miranda",
        "right to counsel", "speedy trial",
    ],
    "730": [
        "sentence", "sentencing", "probation", "parole", "corrections", "prison",
        "incarcerated", "incarceration", "good time", "good conduct", "supervised release",
        "mandatory release", "class 1", "class 2", "class x", "extended term",
        "felony", "misdemeanor", "correctional", "department of corrections",
        "day-for-day", "truth in sentencing", "meritorious good time",
    ],
    "705": ["juvenile", "minor", "delinquent", "youth", "underage"],
    "405": ["mental health", "mental illness", "involuntary", "fitness", "unfit", "competency"],
    "20":  ["expunge", "seal", "expungement", "sealing", "record clearance"],
}

# Step 1 prompt: generate queries without seeing the citation list.
# Corpus description only — no citation strings — so query diversity
# is not anchored to whatever citations happen to be in the prompt.
_QUERY_GENERATION_PROMPT = """\
You are generating a retrieval evaluation dataset for an Illinois criminal law RAG system.

The system's corpus contains:
- ILCS chapters: 720 (criminal offenses), 725 (criminal procedure), 730 (corrections/sentencing),
  705 ILCS 405 (juvenile justice), 405 ILCS 5 (mental health), 20 ILCS 2630 (expungement/sealing)
- Illinois Supreme Court Rules (appellate procedure, discovery, jury selection, filing deadlines)

Generate a JSON array of exactly {n} test cases. Each object must have:
  "id":         sequential string like "q001"
  "query":      a natural language question a criminal justice researcher or advocate might ask
  "corpus":     "ilcs", "iscr", or "out_of_scope"
  "difficulty": "easy" (1 citation, direct lookup), "medium" (2 citations, some synthesis),
                or "hard" (3+ citations, multi-statute reasoning)

Distribution requirements:
- Exactly 5 out_of_scope cases (federal law, civil matters, other states)
- At least 8 iscr cases
- At least 12 hard multi-citation ilcs cases
- Remaining cases: mix of easy and medium ilcs
- Cover all ILCS chapter ranges: 720, 725, 730, 705, 405, 20 ILCS 2630

Query type variety (spread across non-out_of_scope cases):
- ~15% colloquial/layperson ("can you beat a DUI if...", "how long do you stay in jail for...")
- ~15% exact-citation lookups ("what does 730 ILCS 5/3-6-3 say about...")
- ~15% procedure/routing edge cases (queries that could plausibly route to either ilcs or iscr)
- ~55% direct research queries in plain legal language

Respond with ONLY a valid JSON array — no markdown fences, no explanation.
"""

# Step 2 prompt: given fixed queries + full citation list, predict which citations
# a correct retrieval system must return.
_ANNOTATION_PROMPT = """\
You are annotating a retrieval evaluation dataset for an Illinois criminal law RAG system.

For each query below, list the ILCS section citations and/or Illinois Supreme Court Rule numbers
that a correct retrieval system MUST return to fully answer the question. Only use citations
from the exact lists provided — these are the only citations that exist in the database.

Available ILCS section citations (use ONLY these):
{ilcs_citations}

Available Illinois Supreme Court Rule numbers (format as "Rule <number>"):
{iscr_rules}

Queries to annotate:
{queries}

Return a JSON array with one object per query. Each object:
  "id":                   the query id provided
  "expected_citations":   list of citation strings that MUST appear in a correct answer.
                          For out_of_scope queries: empty list [].
                          Be conservative — only include citations directly required.
                          1–4 citations per query is typical.
                          ILCS format: exactly as listed (e.g. "730 ILCS 5/3-6-3")
                          ISCR format: "Rule <number>"
  "notes":                one sentence on what retrieval failure this case would catch

Respond with ONLY a valid JSON array — no markdown fences, no explanation.
"""


def _fetch_all_ilcs_citations(client) -> list[str]:
    """Fetch all distinct ILCS section citations — paginated, no sampling cap."""
    all_cits: set[str] = set()
    page_size = 1000
    offset = 0
    while True:
        rows = (
            client.table("ilcs_chunks")
            .select("section_citation")
            .not_.is_("section_citation", "null")
            .range(offset, offset + page_size - 1)
            .execute()
            .data
        )
        for r in rows:
            if r.get("section_citation"):
                all_cits.add(r["section_citation"])
        if len(rows) < page_size:
            break
        offset += page_size
    return sorted(all_cits)


def _group_citations_by_chapter(all_cits: list[str]) -> dict[str, list[str]]:
    """Group a flat citation list by ILCS chapter prefix (first token before ' ILCS')."""
    by_chapter: dict[str, list[str]] = defaultdict(list)
    for c in all_cits:
        chapter = c.split()[0] if c.split() else "other"
        by_chapter[chapter].append(c)
    return dict(by_chapter)


def _citations_for_batch(
    queries: list[dict],
    citations_by_chapter: dict[str, list[str]],
) -> list[str]:
    """
    Build a focused citation list for an annotation batch.

    Chapters detected as relevant to any query in the batch get their full
    citation list. Undetected chapters get a small fallback sample so the
    model can still recognise cross-chapter edge cases.
    """
    combined = " ".join(q["query"].lower() for q in queries)
    relevant_chapters = {
        chapter
        for chapter, keywords in _CHAPTER_KEYWORDS.items()
        if any(kw in combined for kw in keywords)
    }

    result = []
    for chapter, cits in sorted(citations_by_chapter.items()):
        if chapter in relevant_chapters:
            result.extend(cits)  # full list for relevant chapters
        else:
            step = max(1, len(cits) // _FALLBACK_PER_CHAPTER)
            result.extend(cits[::step][:_FALLBACK_PER_CHAPTER])

    return result


def _fetch_iscr_rules(client) -> list[str]:
    rows = (
        client.table("court_rule_chunks")
        .select("rule_number")
        .not_.is_("rule_number", "null")
        .execute()
        .data
    )
    rules = sorted(set(str(r["rule_number"]) for r in rows if r.get("rule_number")))
    return rules[:_MAX_ISCR_RULES]


def _call_claude(ai: anthropic.Anthropic, prompt: str, max_tokens: int = 8192, temperature: float = 0.4) -> list[dict]:
    """Single Claude call with retry on rate limit. Raises if response can't be parsed as JSON."""
    for attempt in range(5):
        try:
            response = ai.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            break
        except anthropic.RateLimitError:
            if attempt == 4:
                raise
            wait = 60 * (attempt + 1)
            print(f"[RateLimit] Waiting {wait}s before retry {attempt + 2}/5...")
            time.sleep(wait)

    if response.stop_reason == "max_tokens":
        print("[Warning] Response hit max_tokens limit — output may be truncated")

    time.sleep(_INTER_CALL_DELAY)

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    return json.loads(raw)


def _generate_queries(ai: anthropic.Anthropic, n: int) -> list[dict]:
    """Step 1: generate query stubs (id, query, corpus, difficulty) without citations."""
    queries: list[dict] = []
    id_counter = 1
    remaining = n
    batch_num = 0

    while remaining > 0:
        batch_n = min(remaining, _QUERY_BATCH_SIZE)
        batch_num += 1
        print(f"[Queries] Batch {batch_num}: generating {batch_n} queries "
              f"({n - remaining + batch_n}/{n} total)...")

        prompt = _QUERY_GENERATION_PROMPT.format(n=batch_n)
        batch = _call_claude(ai, prompt, max_tokens=8192)[:batch_n]  # guard against over-generation

        for case in batch:
            case["id"] = f"q{id_counter:03d}"
            id_counter += 1

        queries.extend(batch)
        remaining -= len(batch)

    return queries


def _annotate_queries(
    ai: anthropic.Anthropic,
    queries: list[dict],
    citations_by_chapter: dict[str, list[str]],
    iscr_rules: list[str],
) -> dict[str, dict]:
    """
    Step 2: for each query, ask Claude which citations must appear in a correct answer.
    Uses chapter-aware citation lists — each batch only sees full citation lists for
    chapters relevant to its queries, keeping prompt size manageable.
    Returns a dict mapping query id → {expected_citations, notes}.
    """
    annotations: dict[str, dict] = {}

    for batch_start in range(0, len(queries), _ANNOTATION_BATCH_SIZE):
        batch = queries[batch_start: batch_start + _ANNOTATION_BATCH_SIZE]
        batch_end = batch_start + len(batch)

        ilcs_cits = _citations_for_batch(batch, citations_by_chapter)
        print(f"[Annotate] Queries {batch_start + 1}–{batch_end} of {len(queries)} "
              f"({len(ilcs_cits)} ILCS citations in prompt)...")

        queries_block = "\n".join(f'{q["id"]}: {q["query"]}' for q in batch)
        prompt = _ANNOTATION_PROMPT.format(
            ilcs_citations="\n".join(ilcs_cits),
            iscr_rules="\n".join(iscr_rules),
            queries=queries_block,
        )

        results = _call_claude(ai, prompt, max_tokens=4096, temperature=0.2)
        for ann in results:
            annotations[ann["id"]] = ann

    return annotations


def _validate_citations(dataset: list[dict], client) -> list[dict]:
    """
    Validates expected_citations against Supabase.
    Strips citations not found in DB; drops a case only if no valid citations remain
    (out_of_scope cases with empty lists are always kept).
    """
    ilcs_needed: set[str] = set()
    iscr_needed: set[str] = set()

    for case in dataset:
        if case.get("corpus") == "out_of_scope":
            continue
        for cit in case.get("expected_citations", []):
            if cit.upper().startswith("RULE "):
                iscr_needed.add(cit[5:].strip())
            else:
                ilcs_needed.add(cit)

    valid_ilcs: set[str] = set()
    if ilcs_needed:
        rows = (
            client.table("ilcs_chunks")
            .select("section_citation")
            .in_("section_citation", list(ilcs_needed))
            .execute()
            .data
        )
        valid_ilcs = {r["section_citation"] for r in rows if r.get("section_citation")}

    valid_iscr: set[str] = set()
    if iscr_needed:
        rows = (
            client.table("court_rule_chunks")
            .select("rule_number")
            .in_("rule_number", list(iscr_needed))
            .execute()
            .data
        )
        valid_iscr = {str(r["rule_number"]) for r in rows if r.get("rule_number") is not None}

    print(f"\n[Validate] Checked {len(ilcs_needed)} ILCS + {len(iscr_needed)} ISCR citations")

    valid_cases: list[dict] = []
    for case in dataset:
        if case.get("corpus") == "out_of_scope":
            valid_cases.append(case)
            continue

        missing = []
        for cit in case.get("expected_citations", []):
            if cit.upper().startswith("RULE "):
                if cit[5:].strip() not in valid_iscr:
                    missing.append(cit)
            else:
                if cit not in valid_ilcs:
                    missing.append(cit)

        if missing:
            case["expected_citations"] = [
                c for c in case["expected_citations"] if c not in missing
            ]
            if case["expected_citations"]:
                print(f"  [{case['id']}] stripped {len(missing)} invalid citations: {missing}")
                valid_cases.append(case)
            else:
                print(f"  [{case['id']}] dropped entirely — no valid citations remain")
        else:
            valid_cases.append(case)

    kept = len(valid_cases)
    dropped = len(dataset) - kept
    if dropped:
        print(f"[Validate] {kept} cases kept, {dropped} dropped")
    else:
        print(f"[Validate] All citations verified ✓  ({kept} cases kept)")

    return valid_cases


def _print_summary(dataset: list[dict]):
    by_diff: dict[str, int] = defaultdict(int)
    by_corpus: dict[str, int] = defaultdict(int)
    by_num_cits: dict[str, int] = defaultdict(int)

    for case in dataset:
        by_diff[case.get("difficulty", "?")] += 1
        by_corpus[case.get("corpus", "?")] += 1
        n = len(case.get("expected_citations", []))
        bucket = str(n) if n <= 3 else "4+"
        by_num_cits[bucket] += 1

    print(f"\n  Total cases:  {len(dataset)}")
    print(f"  By corpus:    {dict(by_corpus)}")
    print(f"  By difficulty:{dict(by_diff)}")
    print(f"  Citation count per case: {dict(sorted(by_num_cits.items()))}")


def generate(n: int) -> list[dict]:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    ai = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Step 1: generate queries (no citation list shown — avoids circularity)
    queries = _generate_queries(ai, n)
    print(f"[Dataset] Generated {len(queries)} queries")

    # Step 2: fetch full citation list and group by chapter for annotation
    print("[Dataset] Fetching citations from Supabase...")
    all_ilcs_cits = _fetch_all_ilcs_citations(supabase)
    citations_by_chapter = _group_citations_by_chapter(all_ilcs_cits)
    iscr_rules = _fetch_iscr_rules(supabase)
    print(f"[Dataset] {len(all_ilcs_cits)} ILCS citations across "
          f"{len(citations_by_chapter)} chapters, {len(iscr_rules)} ISCR rules")

    if not all_ilcs_cits:
        raise RuntimeError("No ILCS citations found in Supabase — is the corpus ingested?")

    # Step 3: annotate queries with chapter-aware citation lists
    annotations = _annotate_queries(ai, queries, citations_by_chapter, iscr_rules)

    # Merge annotations into query stubs
    dataset: list[dict] = []
    for q in queries:
        ann = annotations.get(q["id"], {})
        dataset.append({
            **q,
            "expected_citations": ann.get("expected_citations", []),
            "notes": ann.get("notes", ""),
        })

    return dataset, supabase


def main():
    parser = argparse.ArgumentParser(description="Generate retrieval evaluation dataset")
    parser.add_argument("--n", type=int, default=65, help="Number of test cases to generate")
    parser.add_argument("--output", default=str(DATASET_PATH), help="Output JSON file path")
    args = parser.parse_args()

    dataset, supabase = generate(args.n)
    print(f"\n[Dataset] Generated {len(dataset)} cases (pre-validation)")

    dataset = _validate_citations(dataset, supabase)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"[Dataset] Saved {len(dataset)} validated cases → {out_path}")
    _print_summary(dataset)


if __name__ == "__main__":
    main()
