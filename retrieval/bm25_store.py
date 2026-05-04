import json
import re
from pathlib import Path

import bm25s
from supabase import Client
from llama_index.core.schema import TextNode

from retrieval.config import COLLECTIONS

BM25_CACHE_DIR = Path("data_files/bm25_cache")
_BM25_INDEX_DIR = BM25_CACHE_DIR / "index"
_BM25_CORPUS_PATH = BM25_CACHE_DIR / "corpus.json"
_BM25_META_PATH = BM25_CACHE_DIR / "meta.json"


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    # Preserve statute citations like 5/7-1, 12-3.05 before stripping punctuation
    statute_pattern = re.findall(r'\d+/\d+[\-\.\d]*', text)
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    tokens.extend(statute_pattern)
    return tokens


def _fetch_counts(client: Client) -> dict[str, int]:
    counts = {}
    for col in COLLECTIONS:
        result = (
            client.table(col.table)
            .select("chunk_id", count="exact")
            .limit(0)
            .execute()
        )
        counts[col.id] = result.count or 0
    return counts


def _cache_is_fresh(current_counts: dict[str, int]) -> bool:
    if not _BM25_META_PATH.exists():
        return False
    if not _BM25_INDEX_DIR.exists():
        return False
    if not _BM25_CORPUS_PATH.exists():
        return False
    stored = json.loads(_BM25_META_PATH.read_text())
    return stored == current_counts


class BM25Retriever:
    def __init__(self, client: Client):
        self.chunk_ids: list[str] = []
        self.texts: list[str] = []
        self.enriched_texts: list[str] = []
        self._metadata: list[dict] = []
        self.bm25: bm25s.BM25 | None = None
        self._load(client)

    def _load(self, client: Client):
        print("[BM25] Checking cache...")
        current_counts = _fetch_counts(client)

        if _cache_is_fresh(current_counts):
            print("[BM25] Cache is fresh — loading from disk (mmap)...")
            self._load_from_cache()
        else:
            print("[BM25] Cache stale or missing — rebuilding from Supabase...")
            self._build_from_supabase(client)
            self._save_cache(current_counts)

        print(f"[BM25] Index ready: {len(self.chunk_ids)} chunks")

    def _load_from_cache(self):
        self.bm25 = bm25s.BM25.load(str(_BM25_INDEX_DIR), mmap=True)
        corpus = json.loads(_BM25_CORPUS_PATH.read_text(encoding="utf-8"))
        self.chunk_ids = corpus["chunk_ids"]
        self.texts = corpus["texts"]
        self.enriched_texts = corpus["enriched_texts"]
        self._metadata = corpus["metadata"]

    def _build_from_supabase(self, client: Client):
        def fetch_all(table: str) -> list[dict]:
            rows = []
            page_size = 1000
            offset = 0
            while True:
                batch = (
                    client.table(table)
                    .select("chunk_id, text, enriched_text, display_citation")
                    .not_.is_("text", "null")
                    .range(offset, offset + page_size - 1)
                    .execute()
                    .data
                )
                rows.extend(batch)
                if len(batch) < page_size:
                    break
                offset += page_size
            return rows

        all_rows: list[dict] = []
        for col in COLLECTIONS:
            print(f"[BM25] Fetching {col.table}...")
            all_rows.extend(fetch_all(col.table))

        self.chunk_ids = [r["chunk_id"] for r in all_rows]
        self.texts = [r["text"] for r in all_rows]
        self.enriched_texts = [r.get("enriched_text") or r["text"] for r in all_rows]
        self._metadata = [
            {k: v for k, v in r.items()
             if k not in ("chunk_id", "text", "enriched_text") and v is not None}
            for r in all_rows
        ]

        if not self.chunk_ids:
            print("[BM25] No chunks found — index will be empty.")
            self.bm25 = None
            return

        print(f"[BM25] Tokenizing {len(self.texts)} chunks...")
        corpus_tokens = [_tokenize(t) for t in self.texts]
        self.bm25 = bm25s.BM25()
        self.bm25.index(corpus_tokens, show_progress=False)

    def _save_cache(self, counts: dict[str, int]):
        print("[BM25] Saving cache to disk...")
        BM25_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _BM25_INDEX_DIR.mkdir(parents=True, exist_ok=True)

        if self.bm25 is not None:
            self.bm25.save(str(_BM25_INDEX_DIR))
        _BM25_CORPUS_PATH.write_text(
            json.dumps({
                "chunk_ids": self.chunk_ids,
                "texts": self.texts,
                "enriched_texts": self.enriched_texts,
                "metadata": self._metadata,
            }, ensure_ascii=False),
            encoding="utf-8",
        )
        _BM25_META_PATH.write_text(json.dumps(counts))
        print("[BM25] Cache saved.")

    def retrieve(self, query: str, top_k: int = 20) -> list[TextNode]:
        if not self.chunk_ids or self.bm25 is None:
            return []

        k = min(top_k, len(self.chunk_ids))
        query_tokens = [_tokenize(query)]
        results, scores = self.bm25.retrieve(query_tokens, k=k, show_progress=False)

        nodes = []
        for idx, score in zip(results[0], scores[0]):
            idx = int(idx)
            score = float(score)
            if score <= 0:
                continue
            metadata = {"bm25_score": score, **self._metadata[idx]}
            node = TextNode(
                id_=self.chunk_ids[idx],
                text=self.enriched_texts[idx],
                metadata=metadata,
            )
            nodes.append(node)

        return nodes
