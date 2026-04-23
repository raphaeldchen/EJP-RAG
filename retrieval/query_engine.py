from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.anthropic import Anthropic

from retrieval.indexes import DualFusionRetriever
from retrieval.postprocessor import CrossEncoderReranker, CitationLabelingPostprocessor


_QA_PROMPT = PromptTemplate(
    "You are a legal research assistant specializing in Illinois criminal law.\n"
    "The context below contains numbered source chunks, each prefixed with its "
    "citation label in brackets (e.g. [720 ILCS 5/7-1 — Justifiable Use of Force]).\n\n"
    "Rules:\n"
    "- Answer using ONLY information from the context below.\n"
    "- Begin your response directly with the answer. Do not repeat, list, or "
    "echo the citation labels or chunk headers from the context.\n"
    "- Every sentence that makes a factual or legal claim must end with the citation "
    "label of the chunk it came from, in brackets.\n"
    "- If a sentence draws on multiple chunks, list all relevant labels.\n"
    "- Use only citation labels that appear verbatim in the context. "
    "Do not invent or paraphrase citations.\n"
    "- If the context does not contain enough information to answer, say so.\n\n"
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
        node_postprocessors=[reranker, CitationLabelingPostprocessor()],
        response_mode="compact",
        text_qa_template=_QA_PROMPT,
    )
