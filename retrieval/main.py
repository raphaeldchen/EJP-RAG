from llama_index.llms.anthropic import Anthropic
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

from retrieval.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, OLLAMA_BASE_URL
from retrieval.embeddings import get_embedding_model
from retrieval.indexes import get_supabase_client, build_all_retrievers
from retrieval.query_engine import build_query_engine
from retrieval.bm25_store import BM25Retriever
from retrieval.reflection import reflect, QueryIntent


def build_rag(use_local: bool = False) -> tuple:
    embed_model = get_embedding_model()
    if use_local:
        llm = Ollama(model="llama3.2", base_url=OLLAMA_BASE_URL, request_timeout=120)
    else:
        llm = Anthropic(model=ANTHROPIC_MODEL, api_key=ANTHROPIC_API_KEY, max_tokens=2048)
    Settings.embed_model = embed_model
    Settings.llm = llm

    client = get_supabase_client()
    bm25 = BM25Retriever(client)

    # Warm up reranker at startup so first query isn't slow
    from retrieval.postprocessor import CrossEncoderReranker
    reranker = CrossEncoderReranker(top_n=10, score_threshold=-3.0)
    reranker._get_model()  # force load now

    retrievers = build_all_retrievers(client, bm25)
    engine = build_query_engine(
        ilcs_retriever=retrievers["ilcs"],
        iscr_retriever=retrievers["iscr"],
        llm=llm,
        reranker=reranker,  # pass the warmed instance
    )
    return engine, retrievers


def query(engine, retrievers: dict, question: str, use_local: bool = False) -> str:
    result = reflect(question, use_local=use_local)
    print(f"\n[Reflection] intent={result.intent.value} | {result.reasoning}")

    if result.intent == QueryIntent.OUT_OF_SCOPE:
        return (
            "That question falls outside this system's scope. "
            "I can only answer questions about Illinois criminal law and court procedure."
        )

    # Multi-query: primary = original (better semantic embedding with nomic-embed-text),
    # secondary = rewritten (adds citation keyword signals for BM25 + citation pinning).
    # Citation pinning scans both queries, so citations in the rewrite still get pinned.
    # If reflection produced no rewrite, single-query mode as before.
    if result.rewritten_query:
        print(f"[Reflection] rewritten → '{result.rewritten_query}'")
        for r in retrievers.values():
            r._secondary_query = result.rewritten_query
        primary_query = question
    else:
        primary_query = question

    try:
        response = engine.query(primary_query)
    finally:
        # Always clear secondary query so it doesn't bleed into the next call
        for r in retrievers.values():
            r._secondary_query = None

    # Extract citations from source nodes
    citations = _extract_citations(response)
    answer = str(response)
    if citations:
        answer += "\n\nSources:\n" + "\n".join(f"  • {c}" for c in citations)

    return answer


def _extract_citations(response) -> list[str]:
    citations = []
    seen = set()

    if not hasattr(response, "source_nodes"):
        return citations

    for node_with_score in response.source_nodes:
        meta = node_with_score.node.metadata

        # ILCS citation
        section = meta.get("section_citation")
        if section and section not in seen:
            seen.add(section)
            citations.append(section)

        # ISCR citation
        rule = meta.get("rule_number")
        title = meta.get("rule_title", "")
        if rule:
            label = f"Rule {rule}" + (f" — {title}" if title else "")
            if label not in seen:
                seen.add(label)
                citations.append(label)

    return citations


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Use local Ollama model instead of Anthropic API")
    args = parser.parse_args()

    engine, retrievers = build_rag(use_local=args.local)

    test_queries = [
        # --- Deterministic rewrite rules (should trigger fast-path rewrites) ---
        "Can a juvenile be tried as an adult for aggravated criminal sexual assault in Illinois?",  # → 705 ILCS 405/5-805

        # --- Complex multi-statute reasoning (tests synthesis across chunks) ---
        "What are the sentencing ranges for a Class 1 felony in Illinois, and under what circumstances can a judge impose an extended-term sentence?",

        # --- Directly relevant to nonprofit mission: incarcerated people & sentence reduction ---
        "How does good-time credit work in Illinois, and how does participation in educational or vocational programs affect sentence length or parole eligibility?",

        # --- Post-conviction relief (important for criminal justice research) ---
        "What post-conviction remedies are available to a defendant in Illinois who claims they received ineffective assistance of counsel?",

        # --- Scope rejection (should be blocked as out-of-scope) ---
        "What are the federal sentencing guidelines for drug trafficking?",

        # --- HEP Demo Queries ---
        "What education or vocational programs are incarcerated individuals entitled to under Illinois law?",
       
        "Can a judge impose education requirements as a condition of probation in Illinois?"
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print("-" * 60)
        print(query(engine, retrievers, q, use_local=args.local))