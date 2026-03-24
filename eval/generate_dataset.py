"""
Generates a golden retrieval evaluation dataset.

Process:
  1. Queries Supabase for the actual citations present in the corpus
  2. Samples diverse citations across chapters/rule ranges
  3. Calls Claude to generate realistic test queries for each citation (and multi-citation combos)
  4. Saves to eval/dataset.json

Usage:
    python -m eval.generate_dataset
    python -m eval.generate_dataset --n 80 --output eval/dataset.json
"""

import json
import argparse
import sys
import random
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

# Max citations per ILCS chapter to include in the prompt (keeps prompt size manageable)
_MAX_PER_CHAPTER = 18
# Max ISCR rule numbers to include
_MAX_ISCR_RULES = 40

_GENERATION_PROMPT = """\
You are generating a retrieval evaluation dataset for an Illinois criminal law RAG system.

The system's corpus contains:
- ILCS chapters: 720 (criminal offenses), 725 (criminal procedure), 730 (corrections/sentencing),
  705 ILCS 405 (juvenile justice), 405 ILCS 5 (mental health), 20 ILCS 2630 (expungement/sealing)
- Illinois Supreme Court Rules (appellate procedure, discovery, jury selection, filing deadlines)

IMPORTANT: Only use citations from the exact lists below — these are the citations that exist
in the database. Do NOT invent citations.

Available ILCS section citations:
{ilcs_citations}

Available Illinois Supreme Court Rule numbers (format as "Rule <number>" in expected_citations):
{iscr_rules}

Generate a JSON array of exactly {n} test cases. Each object must have:
  "id":                   sequential string like "q001"
  "query":                a natural language question a criminal justice researcher or advocate might ask
  "corpus":               "ilcs", "iscr", or "out_of_scope"
  "expected_citations":   for ilcs/iscr: list of citation strings that MUST appear in retrieved results
                          for out_of_scope: empty list []
                          ILCS format: exactly as listed above (e.g. "730 ILCS 5/3-6-3")
                          ISCR format: "Rule <number>" (e.g. "Rule 412")
  "difficulty":           "easy" (1 citation, direct lookup), "medium" (2 citations, some synthesis),
                          or "hard" (3+ citations, multi-statute reasoning)
  "notes":                one sentence on what retrieval failure this case would catch

Distribution requirements:
- Exactly 5 out_of_scope cases (federal law, civil matters, other states) — expected_citations: []
- At least 8 iscr cases
- At least 12 hard multi-citation ilcs cases
- Remaining cases: mix of easy and medium ilcs
- Cover all ILCS chapter ranges: 720, 725, 730, 705, 405, 20 ILCS 2630

Query type variety (spread across the non-out_of_scope cases):
- ~15% colloquial/layperson ("can you beat a DUI if...", "how long do you stay in jail for...")
- ~15% exact-citation lookups ("what does 730 ILCS 5/3-6-3 say about...")
- ~15% procedure/routing edge cases (queries that could plausibly route to either ilcs or iscr)
- ~55% direct research queries in plain legal language

Respond with ONLY a valid JSON array — no markdown fences, no explanation.
"""


def _sample_ilcs_citations(client) -> list[str]:
    """
    Fetch all distinct section_citations and sample up to _MAX_PER_CHAPTER per chapter.
    Groups by leading chapter number (720, 725, 730, etc.).
    """
    rows = (
        client.table("ilcs_chunks")
        .select("section_citation")
        .not_.is_("section_citation", "null")
        .execute()
        .data
    )
    all_cits = sorted(set(r["section_citation"] for r in rows if r.get("section_citation")))

    # Group by chapter prefix (first token before " ILCS")
    by_chapter: dict[str, list[str]] = defaultdict(list)
    for c in all_cits:
        parts = c.split()
        chapter = parts[0] if parts else "other"
        by_chapter[chapter].append(c)

    sampled = []
    for chapter, cits in sorted(by_chapter.items()):
        if len(cits) <= _MAX_PER_CHAPTER:
            sampled.extend(cits)
        else:
            # Evenly spaced sample to capture spread within the chapter
            step = len(cits) // _MAX_PER_CHAPTER
            sampled.extend(cits[::step][:_MAX_PER_CHAPTER])

    return sampled


def _sample_iscr_rules(client) -> list[str]:
    rows = (
        client.table("court_rule_chunks")
        .select("rule_number")
        .not_.is_("rule_number", "null")
        .execute()
        .data
    )
    rules = sorted(set(str(r["rule_number"]) for r in rows if r.get("rule_number")))
    return rules[:_MAX_ISCR_RULES]


def generate(n: int) -> list[dict]:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    print("[Dataset] Fetching citations from Supabase...")
    ilcs_cits = _sample_ilcs_citations(supabase)
    iscr_rules = _sample_iscr_rules(supabase)
    print(f"[Dataset] {len(ilcs_cits)} ILCS citations, {len(iscr_rules)} ISCR rules sampled")

    prompt = _GENERATION_PROMPT.format(
        ilcs_citations="\n".join(ilcs_cits),
        iscr_rules="\n".join(iscr_rules),
        n=n,
    )

    print(f"[Dataset] Calling Claude ({ANTHROPIC_MODEL}) to generate {n} cases...")
    ai = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = ai.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=8192,
        temperature=0.4,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    dataset = json.loads(raw)
    return dataset


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


def main():
    parser = argparse.ArgumentParser(description="Generate retrieval evaluation dataset")
    parser.add_argument("--n", type=int, default=65, help="Number of test cases to generate")
    parser.add_argument("--output", default=str(DATASET_PATH), help="Output JSON file path")
    args = parser.parse_args()

    dataset = generate(args.n)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\n[Dataset] Saved {len(dataset)} cases → {out_path}")
    _print_summary(dataset)


if __name__ == "__main__":
    main()
