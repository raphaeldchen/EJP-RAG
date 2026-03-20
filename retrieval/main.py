from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

from retrieval.config import OLLAMA_BASE_URL
from retrieval.embeddings import get_embedding_model
from retrieval.indexes import get_supabase_client, build_all_retrievers
from retrieval.query_engine import build_query_engine
from retrieval.bm25_store import BM25Retriever
from retrieval.reflection import reflect, QueryIntent


def build_rag(llm_model: str = "llama3.2") -> tuple:
    embed_model = get_embedding_model()
    llm = Ollama(model=llm_model, base_url=OLLAMA_BASE_URL, request_timeout=120.0)
    Settings.embed_model = embed_model
    Settings.llm = llm

    client = get_supabase_client()
    bm25 = BM25Retriever(client)

    # Warm up reranker at startup so first query isn't slow
    from retrieval.postprocessor import CrossEncoderReranker
    reranker = CrossEncoderReranker(top_n=6, score_threshold=0.1)
    reranker._get_model()  # force load now

    retrievers = build_all_retrievers(client, bm25)
    engine = build_query_engine(
        ilcs_retriever=retrievers["ilcs"],
        iscr_retriever=retrievers["iscr"],
        llm=llm,
        reranker=reranker,  # pass the warmed instance
    )
    return engine, llm_model


def query(engine, question: str, llm_model: str) -> str:
    result = reflect(question, model=llm_model)
    print(f"\n[Reflection] intent={result.intent.value} | {result.reasoning}")

    if result.intent == QueryIntent.OUT_OF_SCOPE:
        return (
            "That question falls outside this system's scope. "
            "I can only answer questions about Illinois criminal law and court procedure."
        )

    effective_query = result.rewritten_query or question
    if result.rewritten_query:
        print(f"[Reflection] rewritten → '{result.rewritten_query}'")

    response = engine.query(effective_query)

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
    engine, llm_model = build_rag()

    test_queries = [
        "What are the legal standards for a self-defense claim in Illinois?",
        "What happens if someone violates probation?",
        "Can a juvenile be tried as an adult for aggravated criminal sexual assault in Illinois?",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print("-" * 60)
        print(query(engine, q, llm_model))