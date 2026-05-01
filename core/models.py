from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Chunk:
    chunk_id: str
    parent_id: str
    chunk_index: int
    chunk_total: int
    text: str
    enriched_text: str
    source: str
    token_count: int
    display_citation: str
    metadata: dict = field(default_factory=dict)
    chunked_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
