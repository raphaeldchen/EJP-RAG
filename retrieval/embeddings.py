from llama_index.embeddings.ollama import OllamaEmbedding
from retrieval.config import OLLAMA_BASE_URL, EMBED_MODEL, EMBED_DIM

def get_embedding_model() -> OllamaEmbedding:
    return OllamaEmbedding(
        model_name=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
        embed_batch_size=10,
    )

def embed_query(text: str, model: OllamaEmbedding) -> list[float]:
    embedding = model.get_query_embedding(text)
    if len(embedding) != EMBED_DIM:
        raise ValueError(
            f"Embedding dimension mismatch: expected {EMBED_DIM}, got {len(embedding)}"
        )
    return embedding