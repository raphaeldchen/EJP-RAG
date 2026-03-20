import os
from dotenv import load_dotenv

load_dotenv()

# Supabase
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768

# Retrieval
DEFAULT_TOP_K = 10
SIMILARITY_THRESHOLD = 0.5

# Table / function names
ILCS_RPC = "match_ilcs_chunks"
ISCR_RPC = "match_court_rule_chunks"