from supabase import create_client
from retrieval.config import SUPABASE_URL, SUPABASE_SERVICE_KEY, ILCS_RPC
from retrieval.embeddings import get_embedding_model, embed_query

client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
model = get_embedding_model()

test_vec = embed_query("what is the penalty for aggravated battery", model)

result = client.rpc(ILCS_RPC, {
    "query_embedding": test_vec,
    "match_count": 3,
}).execute()

for row in result.data:
    print(row["chunk_id"], round(row["similarity"], 4))
    print(row["enriched_text"][:120])
    print("---")