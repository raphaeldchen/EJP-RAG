CREATE TABLE IF NOT EXISTS audit_feedback (
    id              uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at      timestamptz DEFAULT now(),
    query_text      text NOT NULL,
    query_id        text NOT NULL,     -- sha256(query_text) for grouping
    chunk_id        text NOT NULL,
    citation        text,
    source          text,              -- "ilcs" | "iscr" | "opinions" | "regulations" | "documents"
    retrieval_mode  text NOT NULL,     -- "hybrid" | "vector" | "bm25"
    persona         text,              -- "researcher" | "practitioner" | "incarcerated"
    pre_rerank_rank integer,           -- position in raw candidates list (1-indexed)
    post_rerank_rank integer,          -- position after reranking, null if dropped
    rrf_score       float,
    ce_score        float,
    label           text NOT NULL      -- "BINDING" | "RELEVANT" | "IRRELEVANT"
                    CHECK (label IN ('BINDING', 'RELEVANT', 'IRRELEVANT')),
    comment         text,
    expert_id       text               -- who submitted (email or name, optional)
);

CREATE INDEX IF NOT EXISTS audit_feedback_query_id_idx ON audit_feedback (query_id);
CREATE INDEX IF NOT EXISTS audit_feedback_label_idx ON audit_feedback (label);
CREATE INDEX IF NOT EXISTS audit_feedback_chunk_id_idx ON audit_feedback (chunk_id);
