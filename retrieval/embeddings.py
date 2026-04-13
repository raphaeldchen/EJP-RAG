from retrieval.config import EMBED_BACKEND, EMBED_MODEL, EMBED_DIM, OLLAMA_BASE_URL


def get_embedding_model():
    """
    Returns an embedding model compatible with the llama_index retriever stack.
    Controlled by EMBED_BACKEND env var:
      "ollama"               — OllamaEmbedding (default, requires Ollama running locally)
      "sentence-transformers" — HuggingFaceEmbedding (downloads model on first use)
    """
    if EMBED_BACKEND == "sentence-transformers":
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        return HuggingFaceEmbedding(model_name=EMBED_MODEL, embed_batch_size=32)
    else:
        from llama_index.embeddings.ollama import OllamaEmbedding
        return OllamaEmbedding(
            model_name=EMBED_MODEL,
            base_url=OLLAMA_BASE_URL,
            embed_batch_size=10,
        )


def embed_query(text: str, model) -> list[float]:
    embedding = model.get_query_embedding(text)
    if len(embedding) != EMBED_DIM:
        raise ValueError(
            f"Embedding dimension mismatch: expected {EMBED_DIM}, got {len(embedding)}"
        )
    return embedding
