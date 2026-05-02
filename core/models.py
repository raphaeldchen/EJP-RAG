from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Chunk:
    """Subdivided segment of an Entry, sized for embedding and retrieval.

    Field conventions:
    - ``parent_id``: equals the ``id`` field of the source Entry (ingest record).
    - ``display_citation``: formatted source label shown in answers and UI.
      For statute/rule sources use the legal citation form, e.g.
      ``"720 ILCS 5/7-1 — Justifiable Use of Force"`` or ``"Rule 431 — Voir Dire"``.
      For document sources (SPAC, ICCB, IDOC, Federal, etc.) use a descriptive
      title, e.g. ``"SPAC Policy Report (2023): Sentencing Reform"``.
      Leave as ``""`` when no meaningful label is available.
    - ``chunked_at``: ISO-8601 UTC timestamp; auto-populated at construction.
      Callers should not pass this field.
    """

    chunk_id: str
    parent_id: str
    chunk_index: int
    chunk_total: int
    text: str
    enriched_text: str
    source: str
    token_count: int
    display_citation: str = ""
    metadata: dict[str, object] = field(default_factory=dict)
    chunked_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
