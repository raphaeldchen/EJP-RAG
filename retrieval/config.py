import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# Supabase
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

# Ollama (used as one possible embedding backend)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Embedding model config — override via env to switch models without code changes.
# EMBED_BACKEND: "ollama" (default) | "sentence-transformers"
# Example for bge-base experiment:
#   EMBED_BACKEND=sentence-transformers EMBED_MODEL=BAAI/bge-base-en-v1.5
#   ILCS_TABLE=ilcs_chunks_bge_base ILCS_RPC=match_ilcs_chunks_bge_base
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "ollama")
EMBED_MODEL   = os.getenv("EMBED_MODEL", "nomic-embed-text")
EMBED_DIM     = int(os.getenv("EMBED_DIM", "768"))

# Anthropic (used for LLM inference and reflection)
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
ANTHROPIC_MODEL = "claude-sonnet-4-6"

# Retrieval
DEFAULT_TOP_K = 40
SIMILARITY_THRESHOLD = 0.5

# Table / RPC names — override via env to point at a different embedding experiment.
ILCS_TABLE = os.getenv("ILCS_TABLE", "ilcs_chunks")
ISCR_TABLE = os.getenv("ISCR_TABLE", "court_rule_chunks")
ILCS_RPC   = os.getenv("ILCS_RPC",   "match_ilcs_chunks")
ISCR_RPC   = os.getenv("ISCR_RPC",   "match_court_rule_chunks")


@dataclass(frozen=True)
class CollectionConfig:
    id: str    # "ilcs" | "iscr" | "opinions" | "regulations" | "documents"
    table: str # Supabase table name
    rpc: str   # vector search RPC function name


COLLECTIONS: list[CollectionConfig] = [
    CollectionConfig("ilcs",        ILCS_TABLE,             ILCS_RPC),
    CollectionConfig("iscr",        ISCR_TABLE,             ISCR_RPC),
    CollectionConfig("opinions",    "opinion_chunks",        "match_opinion_chunks"),
    CollectionConfig("regulations", "regulation_chunks",     "match_regulation_chunks"),
    CollectionConfig("documents",   "document_chunks",       "match_document_chunks"),
]