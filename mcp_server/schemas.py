from pydantic import BaseModel


class ChunkResult(BaseModel):
    chunk_id: str
    text: str
    citation: str
    source: str
    rrf_score: float
    metadata: dict


class SearchResponse(BaseModel):
    query: str
    rewritten_query: str | None
    intent: str
    results: list[ChunkResult]


class LookupResponse(BaseModel):
    citation: str
    chunks: list[ChunkResult]
    total_found: int


class AuditCandidate(BaseModel):
    chunk_id: str
    text: str
    citation: str
    source: str
    rrf_score: float
    ce_score: float | None
    survived: bool
    metadata: dict


class AuditResponse(BaseModel):
    query: str
    rewritten_query: str | None
    intent: str
    retrieval_mode: str          # "hybrid" | "vector" | "bm25"
    candidates: list[AuditCandidate]
    reranked: list[AuditCandidate]
    dropped: list[AuditCandidate]
    threshold: float
    top_n: int


class ClassifyResponse(BaseModel):
    intent: str
    reasoning: str
    rewritten_query: str | None
