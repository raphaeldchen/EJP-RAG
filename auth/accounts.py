import bcrypt
from datetime import datetime, timezone
from supabase import create_client, Client
from retrieval.config import SUPABASE_URL, SUPABASE_SERVICE_KEY


_DUMMY_HASH = bcrypt.hashpw(b"dummy", bcrypt.gensalt()).decode()


def _client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def signup(email: str, password: str) -> tuple[bool, str]:
    email = email.lower().strip()
    client = _client()
    existing = client.table("lawyer_accounts").select("id").eq("email", email).execute()
    if existing.data:
        return False, "An account with that email already exists."
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    client.table("lawyer_accounts").insert({
        "email": email,
        "password_hash": password_hash,
        "approved": False,
    }).execute()
    return True, "Account created."


def login(email: str, password: str) -> tuple[bool, str]:
    email = email.lower().strip()
    rows = _client().table("lawyer_accounts").select("email,password_hash,approved").eq("email", email).execute()
    row = rows.data[0] if rows.data else None
    stored_hash = row["password_hash"].encode() if row else _DUMMY_HASH.encode()
    password_ok = bcrypt.checkpw(password.encode(), stored_hash)
    if not row or not password_ok:
        return False, "Invalid email or password."
    # Intentional: reveals to the user that their email+password are correct but account is pending.
    if not row["approved"]:
        return False, "Your account is pending approval. You'll receive an email when it's ready."
    return True, "ok"


def list_accounts() -> list[dict]:
    return (
        _client()
        .table("lawyer_accounts")
        .select("id,email,approved,created_at,approved_at")
        .order("created_at")
        .execute()
        .data
    )


def approve_account(account_id: str) -> None:
    _client().table("lawyer_accounts").update({
        "approved": True,
        "approved_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", account_id).execute()
