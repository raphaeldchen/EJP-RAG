"""
Export audit_feedback ratings as sentence-transformers training triplets.

Usage:
    python3 scripts/export_triplets.py --output data_files/triplets.jsonl
"""
import argparse
import json
import sys
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

from retrieval.indexes import get_supabase_client
from retrieval.config import ILCS_TABLE, ISCR_TABLE

LABEL_POSITIVE = {"BINDING", "RELEVANT"}
LABEL_NEGATIVE = {"IRRELEVANT"}

_TABLE_MAP = {
    "ilcs": ILCS_TABLE,
    "iscr": ISCR_TABLE,
    "opinions": "opinion_chunks",
    "regulations": "regulation_chunks",
    "documents": "document_chunks",
}


def load_feedback(client) -> list[dict]:
    rows = client.table("audit_feedback").select("*").execute().data
    print(f"Loaded {len(rows)} feedback rows", file=sys.stderr)
    return rows


def fetch_chunk_text(client, chunk_id: str, source: str) -> str | None:
    table = _TABLE_MAP.get(source)
    if not table:
        return None
    try:
        rows = (
            client.table(table)
            .select("enriched_text, text")
            .eq("chunk_id", chunk_id)
            .execute()
            .data
        )
        if rows:
            return rows[0].get("enriched_text") or rows[0].get("text")
    except Exception:
        pass
    return None


def build_triplets(feedback: list[dict], client) -> list[dict]:
    """
    One triplet per query: best positive (highest ce_score among BINDING/RELEVANT) paired
    with hardest negative (highest ce_score among IRRELEVANT). One-best × one-hard keeps
    the training set diverse without generating near-identical triplets from a single query.
    """
    by_query: dict[str, dict] = defaultdict(lambda: {"positives": [], "negatives": []})

    for row in feedback:
        label = row.get("label", "")
        entry = {
            "chunk_id": row["chunk_id"],
            "source": row.get("source", ""),
            "ce_score": row.get("ce_score") or 0.0,
        }
        if label in LABEL_POSITIVE:
            by_query[row["query_text"]]["positives"].append(entry)
        elif label in LABEL_NEGATIVE:
            by_query[row["query_text"]]["negatives"].append(entry)

    triplets = []
    skipped = 0

    for query, data in by_query.items():
        positives = sorted(data["positives"], key=lambda x: x["ce_score"], reverse=True)
        negatives = sorted(data["negatives"], key=lambda x: x["ce_score"], reverse=True)

        if not positives or not negatives:
            skipped += 1
            continue

        best_pos = positives[0]
        hard_neg = negatives[0]

        pos_text = fetch_chunk_text(client, best_pos["chunk_id"], best_pos["source"])
        neg_text = fetch_chunk_text(client, hard_neg["chunk_id"], hard_neg["source"])

        if not pos_text or not neg_text:
            skipped += 1
            continue

        triplets.append({
            "query": query,
            "positive": pos_text,
            "negative": neg_text,
            "positive_chunk_id": best_pos["chunk_id"],
            "negative_chunk_id": hard_neg["chunk_id"],
            "negative_ce_score": hard_neg["ce_score"],
        })

    print(f"Generated {len(triplets)} triplets ({skipped} queries skipped)", file=sys.stderr)
    return triplets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data_files/triplets.jsonl")
    args = parser.parse_args()

    client = get_supabase_client()
    feedback = load_feedback(client)
    triplets = build_triplets(feedback, client)

    with open(args.output, "w") as f:
        for t in triplets:
            f.write(json.dumps(t) + "\n")

    print(f"Wrote {len(triplets)} triplets to {args.output}")


if __name__ == "__main__":
    main()
