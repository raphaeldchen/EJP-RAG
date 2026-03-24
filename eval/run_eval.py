"""
Retrieval evaluation runner.

Tests retrieval at each pipeline stage:
  reflected — Full pipeline: reflection/query-rewriting → reranked (DEFAULT, production config)
  reranked  — RRF fusion + CrossEncoder reranking, no query rewriting (diagnostic baseline)
  fused     — RRF fusion of BM25 + vector, pre-reranker (component debug)
  vector    — Dense vector search only (component debug)
  bm25      — BM25 lexical search only (component debug)

The default run (reranked + reflected) shows both production accuracy and the value added by
query rewriting. Use --no-reflection to drop the reflected stage. Use bm25/vector/fused only
when debugging retrieval components, not for reporting system performance.

Also measures scope accuracy on out_of_scope cases (reflection reject rate) with --scope.

Metrics reported (per stage, macro-averaged across queries):
  Hit@k         at k = 3, 6, 10
  Precision@k   at k = 3, 6, 10
  Recall@k      at k = 3, 6, 10
  MRR           (over full ranked list)
  MAP           (Mean Average Precision)
  nDCG@k        at k = 6, 10

Usage:
    python -m eval.run_eval                                      # default: reranked + reflected
    python -m eval.run_eval --no-reflection                      # raw retrieval only
    python -m eval.run_eval --stages bm25 fused reranked reflected  # all stages
    python -m eval.run_eval --filter-difficulty hard --failures  # drill into hard cases
"""

import json
import argparse
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

from llama_index.core import Settings
from llama_index.core.schema import QueryBundle, NodeWithScore, TextNode

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL
from retrieval.embeddings import get_embedding_model
from retrieval.indexes import get_supabase_client, build_all_retrievers
from retrieval.bm25_store import BM25Retriever
from retrieval.postprocessor import CrossEncoderReranker
from retrieval.reflection import reflect, QueryIntent
from eval.metrics import hit_at_k, precision_at_k, recall_at_k, mrr, ndcg_at_k, average_precision

DATASET_PATH = Path(__file__).parent.parent / "data_files" / "eval_files" / "dataset.json"
K_VALUES = [3, 6, 10]


# ---------------------------------------------------------------------------
# Citation extraction helpers
# ---------------------------------------------------------------------------

def _extract_citation(node) -> str | None:
    """Return a canonical citation string from a NodeWithScore or TextNode."""
    meta = node.node.metadata if isinstance(node, NodeWithScore) else node.metadata
    sec = meta.get("section_citation")
    if sec:
        return sec
    rule = meta.get("rule_number")
    if rule:
        return f"Rule {rule}"
    return None


def _dedup_citations(raw: list[str | None]) -> list[str]:
    """Remove None, deduplicate preserving first-occurrence rank order."""
    seen: set[str] = set()
    out: list[str] = []
    for c in raw:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# Per-stage result accumulator
# ---------------------------------------------------------------------------

@dataclass
class StageResults:
    stage: str
    per_query: list[dict] = field(default_factory=list)
    _sums: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    _count: int = 0

    def record(self, case_id: str, query: str, retrieved: list[str], relevant: set[str]):
        row: dict = {
            "id": case_id,
            "query": query,
            "retrieved_top6": retrieved[:6],
            "relevant": sorted(relevant),
            "n_retrieved": len(retrieved),
        }
        for k in K_VALUES:
            row[f"hit@{k}"]   = hit_at_k(retrieved, relevant, k)
            row[f"p@{k}"]     = precision_at_k(retrieved, relevant, k)
            row[f"r@{k}"]     = recall_at_k(retrieved, relevant, k)
            row[f"ndcg@{k}"]  = ndcg_at_k(retrieved, relevant, k)
            self._sums[f"hit@{k}"]  += row[f"hit@{k}"]
            self._sums[f"p@{k}"]    += row[f"p@{k}"]
            self._sums[f"r@{k}"]    += row[f"r@{k}"]
            self._sums[f"ndcg@{k}"] += row[f"ndcg@{k}"]

        row["mrr"] = mrr(retrieved, relevant)
        row["ap"]  = average_precision(retrieved, relevant)
        self._sums["mrr"] += row["mrr"]
        self._sums["ap"]  += row["ap"]
        self._count += 1
        self.per_query.append(row)

    def averages(self) -> dict[str, float]:
        if self._count == 0:
            return {}
        return {k: v / self._count for k, v in self._sums.items()}

    @property
    def n(self) -> int:
        return self._count


# ---------------------------------------------------------------------------
# Component construction
# ---------------------------------------------------------------------------

def build_components() -> dict:
    print("[Eval] Initialising embedding model...")
    embed_model = get_embedding_model()
    Settings.embed_model = embed_model

    client = get_supabase_client()

    print("[Eval] Building BM25 index (loads all chunks from Supabase)...")
    bm25 = BM25Retriever(client)

    print("[Eval] Loading CrossEncoder reranker...")
    reranker = CrossEncoderReranker(top_n=6, score_threshold=-3.0)
    reranker._get_model()

    print("[Eval] Building vector indexes...")
    retrievers = build_all_retrievers(client, bm25)

    return {
        "bm25": bm25,
        "ilcs_retriever": retrievers["ilcs"],
        "iscr_retriever": retrievers["iscr"],
        "reranker": reranker,
    }


# ---------------------------------------------------------------------------
# Per-stage retrieval
# ---------------------------------------------------------------------------

def retrieve_citations(
    components: dict,
    query: str,
    corpus: str,
    stage: str,
) -> list[str]:
    """
    Run one retrieval stage and return deduplicated, ranked citation strings.

    corpus: "ilcs" | "iscr"
    stage:  "bm25" | "vector" | "fused" | "reranked"
    """
    retriever = components.get(f"{corpus}_retriever")
    bm25: BM25Retriever = components["bm25"]
    reranker: CrossEncoderReranker = components["reranker"]
    qb = QueryBundle(query_str=query)

    if stage == "bm25":
        nodes = bm25.retrieve(query, top_k=20)
        raw = [_extract_citation(n) for n in nodes]

    elif stage == "vector":
        nodes: list[NodeWithScore] = retriever._vector_retriever.retrieve(qb)
        raw = [_extract_citation(n) for n in nodes]

    elif stage == "fused":
        nodes = retriever._retrieve(qb)
        raw = [_extract_citation(n) for n in nodes]

    elif stage == "reranked":
        fused = retriever._retrieve(qb)
        nodes = reranker._postprocess_nodes(fused, qb)
        raw = [_extract_citation(n) for n in nodes]

    else:
        raise ValueError(f"Unknown stage: {stage!r}")

    return _dedup_citations(raw)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

_METRIC_COLS = [
    ("hit@3",   "Hit@3"),
    ("hit@6",   "Hit@6"),
    ("p@3",     "P@3"),
    ("p@6",     "P@6"),
    ("r@3",     "R@3"),
    ("r@6",     "R@6"),
    ("mrr",     "MRR"),
    ("ap",      "MAP"),
    ("ndcg@6",  "nDCG@6"),
    ("ndcg@10", "nDCG@10"),
]

# Benchmark targets for the reflected (production) stage.
# Keyed by metric name; each value is (demo_threshold, refined_threshold).
_TARGETS: dict[str, tuple[float, float]] = {
    "hit@3":   (0.60, 0.78),
    "hit@6":   (0.72, 0.88),
    "p@3":     (0.40, 0.55),
    "p@6":     (0.30, 0.42),
    "r@3":     (0.42, 0.58),
    "r@6":     (0.55, 0.70),
    "mrr":     (0.58, 0.73),
    "ap":      (0.52, 0.68),
    "ndcg@6":  (0.60, 0.75),
    "ndcg@10": (0.58, 0.72),
}


def _target_marker(value: float, demo: float, refined: float) -> str:
    """Return a short status tag showing progress toward benchmarks."""
    if value >= refined:
        return " ✓✓"   # meets refined target
    if value >= demo:
        return "  ✓"   # meets demo target
    gap = demo - value
    return f"-{gap:.2f}"  # how far below demo threshold


def print_summary_table(stage_results: list[StageResults]):
    col_names = [label for _, label in _METRIC_COLS]
    col_w = 9
    stage_w = 22
    header = f"{'Stage':<{stage_w}}" + "".join(f"{h:>{col_w}}" for h in col_names)
    sep = "-" * len(header)

    print("\n" + sep)
    print(header)
    print(sep)

    for sr in stage_results:
        avg = sr.averages()
        row = f"{sr.stage:<{stage_w}}"
        for key, _ in _METRIC_COLS:
            row += f"{avg.get(key, 0.0):>{col_w}.3f}"
        print(row)

    # Benchmark reference rows
    print(sep)
    demo_row    = f"{'  ·· demo target':<{stage_w}}"
    refined_row = f"{'  ·· refined target':<{stage_w}}"
    for key, _ in _METRIC_COLS:
        d, r = _TARGETS.get(key, (0.0, 0.0))
        demo_row    += f"{d:>{col_w}.2f}"
        refined_row += f"{r:>{col_w}.2f}"
    print(demo_row)
    print(refined_row)
    print(sep)

    # Per-metric status for the last stage (production config)
    last = stage_results[-1]
    last_avg = last.averages()
    if last_avg:
        print(f"\nBenchmark status — stage: {last.stage}")
        for key, label in _METRIC_COLS:
            if key not in _TARGETS:
                continue
            val = last_avg.get(key, 0.0)
            d, r = _TARGETS[key]
            marker = _target_marker(val, d, r)
            bar_filled = int(val * 20)
            bar = "█" * bar_filled + "░" * (20 - bar_filled)
            print(f"  {label:<8} {val:.3f}  [{bar}]  {marker}")


def print_per_query(sr: StageResults, k: int = 6):
    print(f"\nPer-query results — stage: {sr.stage}")
    header = f"  {'ID':<8} {'MRR':>5} {'Hit@6':>6} {'R@6':>6}  Query / Retrieved"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in sr.per_query:
        flag = "✓" if row["hit@6"] else "✗"
        print(
            f"  {row['id']:<8} {row['mrr']:>5.2f} {row['hit@6']:>6.0f} {row['r@6']:>6.2f}"
            f"  {flag} {row['query'][:55]}"
        )
        if not row["hit@6"]:
            print(f"           expected: {row['relevant']}")
            print(f"           got:      {row['retrieved_top6']}")


def print_failures(sr: StageResults, top_n: int = 10):
    worst = sorted(sr.per_query, key=lambda x: (x["mrr"], x["r@6"]))[:top_n]
    print(f"\nBottom {len(worst)} queries by MRR — stage: {sr.stage}")
    for row in worst:
        print(f"  [{row['id']}] mrr={row['mrr']:.2f}  {row['query'][:70]}")
        print(f"           expected: {row['relevant']}")
        print(f"           got:      {row['retrieved_top6']}")


def print_scope_accuracy(cases: list[dict]):
    """Evaluate out_of_scope detection via the reflection layer."""
    if not cases:
        return
    print(f"\n[Scope accuracy] Testing {len(cases)} out_of_scope cases via reflection...")
    correct = 0
    for case in cases:
        result = reflect(case["query"])
        hit = result.intent == QueryIntent.OUT_OF_SCOPE
        if hit:
            correct += 1
        else:
            print(f"  MISSED [{case['id']}] classified as {result.intent.value} | {case['query']}")
    print(f"  Scope accuracy: {correct}/{len(cases)} = {correct/len(cases):.1%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run retrieval evaluation")
    parser.add_argument(
        "--dataset", default=str(DATASET_PATH),
        help="Path to dataset JSON (default: eval/dataset.json)",
    )
    parser.add_argument(
        "--stages", nargs="+",
        default=["reranked", "reflected"],
        choices=["bm25", "vector", "fused", "reranked", "reflected"],
        help=(
            "Stages to evaluate. Default: reranked + reflected (production config). "
            "Add bm25/vector/fused for component-level debugging."
        ),
    )
    parser.add_argument(
        "--no-reflection", action="store_true",
        help="Exclude the reflected stage (useful for isolating raw retrieval quality)",
    )
    parser.add_argument(
        "--scope", action="store_true",
        help="Also test out_of_scope rejection accuracy (calls reflection API per case)",
    )
    parser.add_argument(
        "--failures", action="store_true",
        help="Print the 10 worst-performing queries for the last stage",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-query results for every stage",
    )
    parser.add_argument(
        "--filter-corpus", choices=["ilcs", "iscr"],
        help="Only evaluate cases for this corpus",
    )
    parser.add_argument(
        "--filter-difficulty", choices=["easy", "medium", "hard"],
        help="Only evaluate cases of this difficulty",
    )
    parser.add_argument(
        "--output", default=None,
        help="Save detailed results JSON to this path (default: eval/eval_results.json)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"[Eval] Dataset not found: {dataset_path}")
        print("[Eval] Run `python -m eval.generate_dataset` first.")
        sys.exit(1)

    with open(dataset_path) as f:
        dataset: list[dict] = json.load(f)

    # Separate out-of-scope cases
    oos_cases = [c for c in dataset if c.get("corpus") == "out_of_scope"]
    eval_cases = [
        c for c in dataset
        if c.get("corpus") not in ("out_of_scope",) and c.get("expected_citations")
    ]

    # Apply filters
    if args.filter_corpus:
        eval_cases = [c for c in eval_cases if c.get("corpus") == args.filter_corpus]
    if args.filter_difficulty:
        eval_cases = [c for c in eval_cases if c.get("difficulty") == args.filter_difficulty]

    print(
        f"[Eval] Dataset: {len(eval_cases)} retrieval cases"
        + (f" (filtered from {len(dataset)})" if len(eval_cases) < len(dataset) else "")
        + f", {len(oos_cases)} out_of_scope cases"
    )

    if not eval_cases:
        print("[Eval] No cases to evaluate after filtering.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Build retrieval components (slow: BM25 index, model loading)
    # ------------------------------------------------------------------
    components = build_components()
    print()

    # ------------------------------------------------------------------
    # Run each stage
    # ------------------------------------------------------------------
    all_stage_results: list[StageResults] = []

    active_stages = [s for s in args.stages if not (s == "reflected" and args.no_reflection)]

    for stage in active_stages:
        label = "reflected (reflection + rewriting → reranked)" if stage == "reflected" else stage
        print(f"[Eval] Stage: {label} ...")
        sr = StageResults(stage=stage)
        t0 = time.time()

        for case in eval_cases:
            query = case["query"]
            corpus = case["corpus"]

            if stage == "reflected":
                refl = reflect(query)
                if refl.intent == QueryIntent.OUT_OF_SCOPE:
                    retrieved = []
                else:
                    effective = refl.rewritten_query or query
                    retrieved = retrieve_citations(components, effective, corpus, "reranked")
            else:
                retrieved = retrieve_citations(components, query, corpus, stage)

            sr.record(case["id"], query, retrieved, set(case["expected_citations"]))

        elapsed = time.time() - t0
        print(f"       {sr.n} queries in {elapsed:.1f}s  (MRR={sr.averages().get('mrr', 0):.3f})")
        all_stage_results.append(sr)

        if args.verbose:
            print_per_query(sr)

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print_summary_table(all_stage_results)

    # Per-difficulty breakdown for the final stage
    if len(all_stage_results) > 0:
        final = all_stage_results[-1]
        for diff in ("easy", "medium", "hard"):
            subset = [
                c for c in eval_cases if c.get("difficulty") == diff
            ]
            if not subset:
                continue
            sr_diff = StageResults(stage=f"  {diff}")
            for case in subset:
                row = next((r for r in final.per_query if r["id"] == case["id"]), None)
                if row:
                    sr_diff.record(
                        case["id"], case["query"],
                        row["retrieved_top6"],
                        set(case["expected_citations"]),
                    )
            if sr_diff.n:
                print(
                    f"  [{diff:<6} n={sr_diff.n:>2}]  "
                    f"MRR={sr_diff.averages().get('mrr', 0):.3f}  "
                    f"Hit@6={sr_diff.averages().get('hit@6', 0):.3f}  "
                    f"R@6={sr_diff.averages().get('r@6', 0):.3f}  "
                    f"nDCG@6={sr_diff.averages().get('ndcg@6', 0):.3f}"
                )

    # ------------------------------------------------------------------
    # Failures + scope accuracy
    # ------------------------------------------------------------------
    if args.failures and all_stage_results:
        print_failures(all_stage_results[-1])

    if args.scope:
        print_scope_accuracy(oos_cases)

    # ------------------------------------------------------------------
    # Save detailed results
    # ------------------------------------------------------------------
    output_path = Path(args.output) if args.output else Path(__file__).parent.parent / "data_files" / "eval_files" / "eval_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_json = {
        sr.stage: {
            "n_queries": sr.n,
            "averages": sr.averages(),
            "per_query": sr.per_query,
        }
        for sr in all_stage_results
    }
    with open(output_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\n[Eval] Detailed results → {output_path}")


if __name__ == "__main__":
    main()
