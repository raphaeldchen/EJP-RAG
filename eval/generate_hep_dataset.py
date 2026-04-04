"""
Generates a retrieval evaluation dataset for the HEP (Higher Education in Prison)
demo queries defined in retrieval/main.py.

Unlike generate_dataset.py (which samples citations → generates queries), this script
works in reverse: given fixed queries, it asks Claude to predict which ILCS/ISCR citations
a correct retrieval system should return, then validates every predicted citation against
Supabase before saving.

Output: data_files/eval_files/hep_dataset.json
Usage:
    python -m eval.generate_hep_dataset
    python -m eval.generate_hep_dataset --output path/to/hep_dataset.json
"""

import json
import argparse
import sys
from pathlib import Path

import anthropic
from supabase import create_client

sys.path.insert(0, str(Path(__file__).parent.parent))
from retrieval.config import (
    SUPABASE_URL,
    SUPABASE_SERVICE_KEY,
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
)

OUTPUT_PATH = Path(__file__).parent.parent / "data_files" / "eval_files" / "hep_dataset.json"

# The 7 HEP demo queries from retrieval/main.py (out-of-scope query excluded)
HEP_QUERIES = [
    {
        "id": "hep001",
        "query": "What are the conditions for good conduct credit for incarcerated individuals in Illinois?",
        "corpus": "ilcs",
        "difficulty": "medium",
    },
    {
        "id": "hep002",
        "query": "How does good-time credit work in Illinois, and how does participation in educational or vocational programs affect sentence length or parole eligibility?",
        "corpus": "ilcs",
        "difficulty": "hard",
    },
    {
        "id": "hep003",
        "query": "What post-conviction remedies are available to a defendant in Illinois who claims they received ineffective assistance of counsel?",
        "corpus": "ilcs",
        "difficulty": "hard",
    },
    {
        "id": "hep004",
        "query": "What education or vocational programs are incarcerated individuals entitled to under Illinois law?",
        "corpus": "ilcs",
        "difficulty": "medium",
    },
    {
        "id": "hep005",
        "query": "What are the eligibility requirements for mandatory supervised release in Illinois?",
        "corpus": "ilcs",
        "difficulty": "medium",
    },
    {
        "id": "hep006",
        "query": "What does Illinois law say about reducing sentences for program participation?",
        "corpus": "ilcs",
        "difficulty": "medium",
    },
    {
        "id": "hep007",
        "query": "Can a judge impose education requirements as a condition of probation in Illinois?",
        "corpus": "ilcs",
        "difficulty": "easy",
    },
]

_ANNOTATION_PROMPT = """\
You are annotating a retrieval evaluation dataset for an Illinois criminal law RAG system.

For each query below, list the ILCS section citations that a correct retrieval system MUST
return to fully answer the question. Only use citations from the exact list provided —
these are the only citations that exist in the database.

Available ILCS section citations (use ONLY these):
{ilcs_citations}

Available Illinois Supreme Court Rule numbers (format as "Rule <number>"):
{iscr_rules}

Queries to annotate:
{queries}

Return a JSON array with one object per query. Each object:
  "id":                   the query id provided
  "expected_citations":   list of citation strings from the lists above that MUST appear
                          in a correct answer. Be conservative — only include citations
                          that are directly required. 1–4 citations per query is typical.
                          ILCS format: exactly as listed (e.g. "730 ILCS 5/3-6-3")
                          ISCR format: "Rule <number>"
  "notes":                one sentence on what retrieval failure this case would catch

Respond with ONLY a valid JSON array — no markdown fences, no explanation.
"""

_MAX_ISCR_RULES = 40


def _fetch_all_ilcs_citations(client) -> list[str]:
    """Return all distinct ILCS section citations in the corpus — paginated, no cap."""
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


def _annotate_queries(ai: anthropic.Anthropic, ilcs_cits: list[str], iscr_rules: list[str]) -> list[dict]:
    queries_block = "\n".join(
        f'{q["id"]}: {q["query"]}' for q in HEP_QUERIES
    )
    prompt = _ANNOTATION_PROMPT.format(
        ilcs_citations="\n".join(ilcs_cits),
        iscr_rules="\n".join(iscr_rules),
        queries=queries_block,
    )

    response = ai.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=4096,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    return json.loads(raw)


def _validate_citations(dataset: list[dict], client) -> list[dict]:
    ilcs_needed: set[str] = set()
    iscr_needed: set[str] = set()

    for case in dataset:
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

    valid_cases: list[dict] = []
    for case in dataset:
        missing = []
        for cit in case.get("expected_citations", []):
            if cit.upper().startswith("RULE "):
                if cit[5:].strip() not in valid_iscr:
                    missing.append(cit)
            else:
                if cit not in valid_ilcs:
                    missing.append(cit)

        if missing:
            print(f"  [{case['id']}] dropping invalid citations: {missing}")
            # Keep the case but strip the invalid citations rather than dropping the whole case
            case["expected_citations"] = [
                c for c in case["expected_citations"] if c not in missing
            ]
            if case["expected_citations"]:
                valid_cases.append(case)
            else:
                print(f"  [{case['id']}] dropped entirely — no valid citations remain")
        else:
            valid_cases.append(case)

    return valid_cases


def main():
    parser = argparse.ArgumentParser(description="Generate HEP demo eval dataset")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="Output JSON file path")
    args = parser.parse_args()

    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    print("[HEP Dataset] Fetching citations from Supabase...")
    ilcs_cits = _fetch_all_ilcs_citations(supabase)
    iscr_rules = _sample_iscr_rules(supabase)
    print(f"[HEP Dataset] {len(ilcs_cits)} ILCS citations, {len(iscr_rules)} ISCR rules available")

    ai = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    print(f"[HEP Dataset] Asking Claude to annotate {len(HEP_QUERIES)} HEP queries...")
    annotations = _annotate_queries(ai, ilcs_cits, iscr_rules)

    # Merge annotations back into the query metadata
    annotation_by_id = {a["id"]: a for a in annotations}
    dataset: list[dict] = []
    for q in HEP_QUERIES:
        ann = annotation_by_id.get(q["id"], {})
        case = {
            **q,
            "expected_citations": ann.get("expected_citations", []),
            "notes": ann.get("notes", ""),
        }
        dataset.append(case)

    print("\n[HEP Dataset] Validating citations against Supabase...")
    dataset = _validate_citations(dataset, supabase)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\n[HEP Dataset] Saved {len(dataset)} cases → {out_path}")
    print("\nCases:")
    for case in dataset:
        print(f"  [{case['id']}] {case['query'][:60]}")
        print(f"           citations: {case['expected_citations']}")

    print(f"\nTo evaluate: python -m eval.run_eval --dataset {out_path}")


if __name__ == "__main__":
    main()
