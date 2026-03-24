"""
Pure metric functions for retrieval evaluation.

All functions take:
  retrieved: ordered list of citation strings (ranked 1..n)
  relevant:  set of citation strings that are correct for the query
"""

import math


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """1 if any relevant citation appears in the top-k results, else 0."""
    return float(any(c in relevant for c in retrieved[:k]))


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of top-k retrieved citations that are relevant."""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return sum(1 for c in top_k if c in relevant) / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant citations found in top-k retrieved results."""
    if not relevant:
        return 1.0  # vacuously perfect — nothing required
    return sum(1 for c in retrieved[:k] if c in relevant) / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Mean Reciprocal Rank — 1/rank of first relevant result, 0 if none found."""
    for rank, c in enumerate(retrieved, start=1):
        if c in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at k.
    Uses binary relevance (1 if in relevant set, 0 otherwise).
    """
    def dcg(cits: list[str], cutoff: int) -> float:
        return sum(
            (1.0 if c in relevant else 0.0) / math.log2(i + 2)
            for i, c in enumerate(cits[:cutoff])
        )

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0:
        return 1.0  # vacuously perfect — nothing required
    return dcg(retrieved, k) / idcg


def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    """
    Average Precision (AP) — area under the precision-recall curve.
    Used to compute MAP across queries.
    """
    if not relevant:
        return 1.0
    hits = 0
    score = 0.0
    for rank, c in enumerate(retrieved, start=1):
        if c in relevant:
            hits += 1
            score += hits / rank
    return score / len(relevant)
