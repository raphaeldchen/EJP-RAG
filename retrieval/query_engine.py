from llama_index.core.query_engine import RetrieverQueryEngine, RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama

from retrieval.indexes import FusionRetriever
from retrieval.postprocessor import CrossEncoderReranker


_ILCS_DESCRIPTION = (
    "Use this index for questions about Illinois substantive criminal law — "
    "criminal offenses, elements of crimes, affirmative defenses (self-defense, "
    "insanity, necessity), sentencing ranges, definitions, juvenile transfer to "
    "adult court, probation, parole, and the text of Illinois Compiled Statutes. "
    "Always use this index when the query contains an ILCS citation like "
    "720 ILCS, 730 ILCS, or 705 ILCS."
)

_ISCR_DESCRIPTION = (
    "Use this index ONLY for questions that explicitly ask about Illinois Supreme Court "
    "Rule numbers, appellate filing deadlines, notice of appeal requirements, discovery "
    "disclosures, jury selection procedures, or other named court rules. "
    "Do NOT use for questions about criminal offenses, penalties, sentencing, probation, "
    "parole, defenses, or the substance of any Illinois statute. "
    "If the query contains an ILCS citation (e.g. 730 ILCS, 720 ILCS, 705 ILCS), "
    "always route to the other index."
)

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
    ilcs_retriever: FusionRetriever,
    iscr_retriever: FusionRetriever,
    llm: Ollama,
    reranker: CrossEncoderReranker = None,
) -> RouterQueryEngine:
    if reranker is None:
        reranker = CrossEncoderReranker(top_n=6, score_threshold=0.1)

    ilcs_engine = RetrieverQueryEngine.from_args(
        retriever=ilcs_retriever,
        llm=llm,
        node_postprocessors=[reranker],
        response_mode="tree_summarize",
        text_qa_template=_QA_PROMPT,
    )
    iscr_engine = RetrieverQueryEngine.from_args(
        retriever=iscr_retriever,
        llm=llm,
        node_postprocessors=[reranker],
        response_mode="tree_summarize",
        text_qa_template=_QA_PROMPT,
    )

    ilcs_tool = QueryEngineTool.from_defaults(
        query_engine=ilcs_engine,
        description=_ILCS_DESCRIPTION,
    )
    iscr_tool = QueryEngineTool.from_defaults(
        query_engine=iscr_engine,
        description=_ISCR_DESCRIPTION,
    )

    return RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(llm=llm),
        query_engine_tools=[ilcs_tool, iscr_tool],
        verbose=True,
    )