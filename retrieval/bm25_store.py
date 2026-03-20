import re
from supabase import Client
from rank_bm25 import BM25Okapi
from llama_index.core.schema import TextNode


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    # Preserve statute citations like 5/7-1, 12-3.05 before stripping punctuation
    statute_pattern = re.findall(r'\d+/\d+[\-\.\d]*', text)
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    # Re-add statute citations as tokens
    tokens.extend(statute_pattern)
    return tokens


class BM25Retriever:
    def __init__(self, client: Client):
        self.chunk_ids: list[str] = []
        self.texts: list[str] = []
        self.bm25: BM25Okapi | None = None
        self._load(client)

    def _load(self, client: Client):
        print("[BM25] Loading corpus from Supabase...")

        def fetch_all(table: str) -> list[dict]:
            rows = []
            page_size = 1000
            offset = 0
            while True:
                batch = (
                    client.table(table)
                    .select("chunk_id, text")
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

        all_rows = fetch_all("ilcs_chunks") + fetch_all("court_rule_chunks")
        self.chunk_ids = [r["chunk_id"] for r in all_rows]
        self.texts = [r["text"] for r in all_rows]

        tokenized = [_tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)
        print(f"[BM25] Index built: {len(self.chunk_ids)} chunks")

    def retrieve(self, query: str, top_k: int = 20) -> list[TextNode]:
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # Get top_k indices sorted by score descending
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        nodes = []
        for idx in top_indices:
            if scores[idx] == 0:
                continue  # skip zero-score results entirely
            node = TextNode(
                id_=self.chunk_ids[idx],
                text=self.texts[idx],
                metadata={"bm25_score": float(scores[idx])},
            )
            nodes.append(node)

        return nodes