from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.anthropic import Anthropic

from retrieval.indexes import DualFusionRetriever
from retrieval.postprocessor import CrossEncoderReranker


_QA_PROMPT = PromptTemplate(
    "You are a legal research assistant specializing in Illinois criminal law.\n"
    "Using only the context below, answer the question precisely.\n"
    "For each point you make, cite the source in brackets — statute section "
    "for ILCS (e.g. [720 ILCS 5/12-3.05]) or rule number for court rules "
    "(e.g. [Rule 606]).\n\n"
    "Context:\n{context_str}\n\n"
    "Question: {query_str}\n\n"
    "Answer:"
)


def build_query_engine(
    dual_retriever: DualFusionRetriever,
    llm: Anthropic,
    reranker: CrossEncoderReranker = None,
) -> RetrieverQueryEngine:
    if reranker is None:
        reranker = CrossEncoderReranker(top_n=10, score_threshold=-3.0)

    return RetrieverQueryEngine.from_args(
        retriever=dual_retriever,
        llm=llm,
        node_postprocessors=[reranker],
        response_mode="tree_summarize",
        text_qa_template=_QA_PROMPT,
    )
