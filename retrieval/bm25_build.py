"""Force a full BM25 index rebuild from Supabase.

Usage: python3 -m retrieval.bm25_build

Deletes the existing cache and rebuilds from scratch. Run this after
a batch_embed job completes to keep the index current without waiting
for a count-change at the next startup.
"""
import shutil
from dotenv import load_dotenv
from supabase import create_client

from retrieval.bm25_store import BM25_CACHE_DIR, BM25Retriever
from retrieval.config import SUPABASE_URL, SUPABASE_SERVICE_KEY


def main():
    load_dotenv()
    if BM25_CACHE_DIR.exists():
        shutil.rmtree(BM25_CACHE_DIR)
        print(f"[bm25_build] Cleared {BM25_CACHE_DIR}")
    client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    BM25Retriever(client)
    print("[bm25_build] Done.")


if __name__ == "__main__":
    main()
