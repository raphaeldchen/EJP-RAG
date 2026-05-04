-- Migration 001: create opinion_chunks, regulation_chunks, document_chunks
--
-- Run in the Supabase SQL editor. Safe to re-run (all statements are IF NOT EXISTS).
--
-- These three tables use the shared Chunk schema. Source-specific fields live in
-- the metadata JSONB column rather than dedicated columns, unlike the legacy
-- ilcs_chunks and court_rule_chunks tables.

-- pgvector extension (already enabled if ilcs_chunks exists, included for safety)
CREATE EXTENSION IF NOT EXISTS vector;


-- ---------------------------------------------------------------------------
-- opinion_chunks
-- Sources: cap_bulk (~250-350K chunks after criminal filter), courtlistener
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS opinion_chunks (
    chunk_id        text        PRIMARY KEY,
    parent_id       text,
    chunk_index     integer,
    chunk_total     integer,
    source          text,           -- 'cap_bulk' | 'courtlistener'
    token_count     integer,
    display_citation text,
    text            text,
    enriched_text   text,
    metadata        jsonb       DEFAULT '{}',
    embedding       vector(768)
);

CREATE INDEX IF NOT EXISTS opinion_chunks_embedding_idx
    ON opinion_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS opinion_chunks_source_idx
    ON opinion_chunks (source);

CREATE OR REPLACE FUNCTION match_opinion_chunks(
    query_embedding vector(768),
    match_count     int         DEFAULT 40
)
RETURNS TABLE (
    chunk_id        text,
    parent_id       text,
    chunk_index     integer,
    chunk_total     integer,
    source          text,
    token_count     integer,
    display_citation text,
    text            text,
    enriched_text   text,
    metadata        jsonb,
    similarity      float
)
LANGUAGE sql STABLE
AS $$
    SELECT
        chunk_id, parent_id, chunk_index, chunk_total,
        source, token_count, display_citation,
        text, enriched_text, metadata,
        1 - (embedding <=> query_embedding) AS similarity
    FROM opinion_chunks
    ORDER BY embedding <=> query_embedding
    LIMIT match_count;
$$;


-- ---------------------------------------------------------------------------
-- regulation_chunks
-- Sources: iac (~860 chunks), idoc (~2K chunks)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS regulation_chunks (
    chunk_id        text        PRIMARY KEY,
    parent_id       text,
    chunk_index     integer,
    chunk_total     integer,
    source          text,           -- 'iac' | 'idoc'
    token_count     integer,
    display_citation text,
    text            text,
    enriched_text   text,
    metadata        jsonb       DEFAULT '{}',
    embedding       vector(768)
);

CREATE INDEX IF NOT EXISTS regulation_chunks_embedding_idx
    ON regulation_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS regulation_chunks_source_idx
    ON regulation_chunks (source);

CREATE OR REPLACE FUNCTION match_regulation_chunks(
    query_embedding vector(768),
    match_count     int         DEFAULT 40
)
RETURNS TABLE (
    chunk_id        text,
    parent_id       text,
    chunk_index     integer,
    chunk_total     integer,
    source          text,
    token_count     integer,
    display_citation text,
    text            text,
    enriched_text   text,
    metadata        jsonb,
    similarity      float
)
LANGUAGE sql STABLE
AS $$
    SELECT
        chunk_id, parent_id, chunk_index, chunk_total,
        source, token_count, display_citation,
        text, enriched_text, metadata,
        1 - (embedding <=> query_embedding) AS similarity
    FROM regulation_chunks
    ORDER BY embedding <=> query_embedding
    LIMIT match_count;
$$;


-- ---------------------------------------------------------------------------
-- document_chunks
-- Sources: spac (~6K), iccb (~1.2K), federal (~120), restorejustice (~130),
--          cookcounty_pd (~130)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id        text        PRIMARY KEY,
    parent_id       text,
    chunk_index     integer,
    chunk_total     integer,
    source          text,           -- 'spac' | 'iccb' | 'federal' | 'restorejustice' | 'cookcounty_pd'
    token_count     integer,
    display_citation text,
    text            text,
    enriched_text   text,
    metadata        jsonb       DEFAULT '{}',
    embedding       vector(768)
);

CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
    ON document_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS document_chunks_source_idx
    ON document_chunks (source);

CREATE OR REPLACE FUNCTION match_document_chunks(
    query_embedding vector(768),
    match_count     int         DEFAULT 40
)
RETURNS TABLE (
    chunk_id        text,
    parent_id       text,
    chunk_index     integer,
    chunk_total     integer,
    source          text,
    token_count     integer,
    display_citation text,
    text            text,
    enriched_text   text,
    metadata        jsonb,
    similarity      float
)
LANGUAGE sql STABLE
AS $$
    SELECT
        chunk_id, parent_id, chunk_index, chunk_total,
        source, token_count, display_citation,
        text, enriched_text, metadata,
        1 - (embedding <=> query_embedding) AS similarity
    FROM document_chunks
    ORDER BY embedding <=> query_embedding
    LIMIT match_count;
$$;
