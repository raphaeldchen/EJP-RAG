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
        self._metadata: list[dict] = []
        self.bm25: BM25Okapi | None = None
        self._load(client)

    def _load(self, client: Client):
        print("[BM25] Loading corpus from Supabase...")

        def fetch_all(table: str, extra_cols: list[str]) -> list[dict]:
            rows = []
            page_size = 1000
            offset = 0
            cols = ", ".join(["chunk_id", "text"] + extra_cols)
            while True:
                batch = (
                    client.table(table)
                    .select(cols)
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

        ilcs_rows = fetch_all("ilcs_chunks", ["display_citation"])
        iscr_rows = fetch_all("court_rule_chunks", ["display_citation"])
        all_rows = ilcs_rows + iscr_rows
        self.chunk_ids = [r["chunk_id"] for r in all_rows]
        self.texts = [r["text"] for r in all_rows]
        self._metadata = [
            {k: v for k, v in r.items() if k not in ("chunk_id", "text") and v is not None}
            for r in all_rows
        ]

        tokenized = [_tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)
        print(f"[BM25] Index built: {len(self.chunk_ids)} chunks")

    def retrieve(self, query: str, top_k: int = 20) -> list[TextNode]:
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)

        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        nodes = []
        for idx in top_indices:
            if scores[idx] == 0:
                continue  # skip zero-score results entirely
            metadata = {"bm25_score": float(scores[idx]), **self._metadata[idx]}
            node = TextNode(
                id_=self.chunk_ids[idx],
                text=self.texts[idx],
                metadata=metadata,
            )
            nodes.append(node)

        return nodes