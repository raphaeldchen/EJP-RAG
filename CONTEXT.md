# Illinois Legal RAG

A retrieval-augmented generation system for Illinois criminal justice research, with a focus on higher education in prison and reentry. It ingests legal and policy sources, chunks and embeds them, and answers natural-language queries with cited sources.

## Language

### Pipeline data units

**Entry**:
The raw unit produced by an ingest script before chunking. Every source produces Entries; subtypes differ by source.
_Avoid_: Record, document (as a generic), item

**Section**:
An Entry from a statute or rule source — ILCS, Illinois Admin Code, or Illinois Supreme Court Rules. Identified by a section citation or rule number.
_Avoid_: Provision, article (for the unit itself)

**Opinion**:
An Entry from a court case source — CAP bulk download or CourtListener. Represents a single court decision.
_Avoid_: Case (as a stored unit), ruling

**Document**:
An Entry from a report or directive source — SPAC, ICCB, IDOC, Federal, Restore Justice, Cook County PD.
_Avoid_: Report, file (as a stored unit)

**Chunk**:
A subdivided segment of an Entry, sized for embedding and retrieval. Identified by `chunk_id`, with `chunk_index` and `chunk_total` tracking its position within the parent Entry.
_Avoid_: Node (internal llama-index term), passage, fragment, excerpt

**Corpus**:
The full unified body of legal knowledge the system can search — all Collections combined, as stored in Supabase.
_Avoid_: Database, knowledge base, index (as a synonym)

**Collection**:
The set of Entries from a single source (e.g., the ILCS Collection, the IDOC Collection). Each Collection is ingested and chunked independently.
_Avoid_: Source (ambiguous with the `source` metadata field), dataset, corpus (reserve for the unified whole)

**Query Analysis**:
The preprocessing step that classifies a query's intent (in-scope, out-of-scope, or ambiguous) and optionally produces a Rewritten Query before retrieval.
_Avoid_: Reflection (internal code term), query rewriting (names only half the step)

**Rewritten Query**:
The version of a query output by Query Analysis, augmented with explicit statutory citations where the system can identify them. Sent to retrieval alongside or in place of the original.
_Avoid_: Enriched query, secondary query

**Citation Pinning**:
A retrieval behavior that deterministically injects Chunks matching an explicit statute citation found in the query, bypassing normal ranking. Pinned Chunks appear in the result set regardless of retrieval score.
_Avoid_: Citation injection, hardcoded lookup

**Candidate**:
A Chunk returned by initial retrieval (vector search or BM25) that is scored and ranked but not yet reranked. The reranker operates on the Candidate pool.
_Avoid_: Result (reserve for the final reranked output), node (internal llama-index term)

**Citation**:
A formatted legal reference included in a system answer, pointing to a specific Chunk's source — e.g., `[720 ILCS 5/7-1 — Justifiable Use of Force]` or `[Rule 431 — Voir Dire]`.
_Avoid_: Reference, attribution, source (as an output concept)

### Users

**Researcher**:
A professor, policy analyst, or academic studying Illinois criminal justice — asks policy-level questions drawing on SPAC, ICCB, and federal education sources.

**Practitioner**:
A public defender or legal advocate — asks procedural and statutory questions requiring ILCS/ISCR precision and case law.

**Incarcerated Person**:
Someone currently incarcerated seeking to understand their rights, eligibility, or available programs — asks practical questions drawing on IDOC directives, ICCB enrollment data, and federal Pell grant guidance.

### System behavior

**Scope**:
The domain of questions the system answers: Illinois criminal justice, including correctional education, reentry, sentencing policy, and the federal law that intersects with Illinois prisoners. Queries outside this domain are rejected by Query Analysis.
_Avoid_: "Illinois criminal law only" (too narrow — the system intentionally includes federal sources relevant to Illinois corrections)

### Labeling and training

**Audit Dashboard**:
The Streamlit application lawyers use to label retrieved Chunks. Imports retrieval logic directly from the MCP server module — lawyers evaluate the exact code path the LangGraph agent will call.
_Avoid_: Labeling tool, review interface

**Label**:
A lawyer's classification of a single Chunk's relevance to a query. One of: BINDING (direct answer — highest-value training positive), RELEVANT (useful context), IRRELEVANT (noise). A Label is an annotation decision, not user feedback on system quality.
_Avoid_: Rating, feedback (for the classification act itself)

**Feedback Record**:
A row in `audit_feedback` storing one Label, along with the query, Chunk metadata, retrieval scores (RRF and CrossEncoder), retrieval mode, persona, and annotator identity.
_Avoid_: Feedback row, audit row, rating

**Training Triplet**:
A (query, positive Chunk, negative Chunk) tuple derived from Feedback Records for embedding model fine-tuning. One triplet per query: the strongest positive (highest CrossEncoder score among BINDING/RELEVANT labels) paired with the hardest negative (highest CrossEncoder score among IRRELEVANT labels).
_Avoid_: Training pair, training example

**Hard Negative**:
An IRRELEVANT Chunk with a high CrossEncoder score — the reranker was confident but wrong. The most valuable Feedback Records for fine-tuning: they teach the embedding model to distinguish near-misses the reranker cannot.
_Avoid_: False positive (reserve for precision metrics), confuser

## Relationships

- A **Section**, **Opinion**, or **Document** is a subtype of **Entry**
- An **Entry** belongs to exactly one **Collection**
- An **Entry** is produced by exactly one ingest script
- An **Entry** is split into one or more **Chunks**
- A **Chunk** has a `parent_id` referencing its source **Entry** and a positional index within that Entry
- The **Corpus** is the union of all **Collections**

## Example dialogue

> **Dev:** "A query about Pell grant eligibility returned a Citation to an IDOC directive instead of the federal ED guidance — why?"
>
> **Researcher:** "The ED Dear Colleague Letter Entry was split into three Chunks, but the relevant Chunk ranked 8th in the Candidate pool. The reranker dropped it. Citation Pinning didn't trigger because the query had no explicit statute citation."
>
> **Dev:** "So the Chunk was in the pool but scored below threshold — a reranker failure, not a retrieval failure?"
>
> **Researcher:** "Exactly. This is an in-scope query for an Incarcerated Person. The Answer needs to cite federal Pell grant rules, not just IDOC policy. The Collection coverage is fine; the ranking is the problem."

## Pending decisions

- **Per-persona retrieval paths (Month 3)**: Persona is now tracked in every Feedback Record via the Audit Dashboard. Whether to build differentiated retrieval paths (separate prompts, collection weights, or reranker thresholds per persona) is deferred to Month 3. Current handling: lightweight persona dropdown in the Audit Dashboard for data collection only — no retrieval behavior changes yet.

## Flagged ambiguities

- "section" is used both as a domain subtype (an Entry from a statute source) and informally to mean a part of any legal text — when precision matters, use the subtype name **Section**
- `source` (lowercase) in chunk metadata is a string identifier (e.g., `"ilcs"`) pointing to which **Collection** a Chunk belongs to — not a domain term itself
- ~~"Illinois criminal law" as the system's scope~~ — resolved: scope is **Illinois criminal justice** (correctional education, reentry, sentencing policy, and intersecting federal law). CLAUDE.md updated 2026-04-30.
