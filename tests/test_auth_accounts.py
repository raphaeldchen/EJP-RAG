import bcrypt
import pytest
from unittest.mock import patch, MagicMock
from auth.accounts import signup, login, list_accounts, approve_account


def _hashed(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


class TestSignup:
    def test_creates_account_with_normalized_email(self):
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []

        with patch("auth.accounts._client", return_value=mock_client):
            success, msg = signup("Lawyer@Example.com", "password123")

        assert success is True
        inserted = mock_client.table.return_value.insert.call_args[0][0]
        assert inserted["email"] == "lawyer@example.com"
        assert inserted["approved"] is False
        assert bcrypt.checkpw(b"password123", inserted["password_hash"].encode())

    def test_rejects_duplicate_email(self):
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [{"id": "x"}]

        with patch("auth.accounts._client", return_value=mock_client):
            success, msg = signup("existing@example.com", "password123")

        assert success is False
        assert "already exists" in msg


class TestLogin:
    def test_success(self):
        hashed = _hashed("correct")
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [
            {"email": "a@b.com", "password_hash": hashed, "approved": True}
        ]

        with patch("auth.accounts._client", return_value=mock_client):
            success, _ = login("a@b.com", "correct")

        assert success is True

    def test_wrong_password(self):
        hashed = _hashed("correct")
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [
            {"email": "a@b.com", "password_hash": hashed, "approved": True}
        ]

        with patch("auth.accounts._client", return_value=mock_client):
            success, msg = login("a@b.com", "wrong")

        assert success is False
        assert "Invalid email or password" in msg

    def test_pending_approval(self):
        hashed = _hashed("pw")
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [
            {"email": "a@b.com", "password_hash": hashed, "approved": False}
        ]

        with patch("auth.accounts._client", return_value=mock_client):
            success, msg = login("a@b.com", "pw")

        assert success is False
        assert "pending" in msg.lower()

    def test_no_account(self):
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []

        with patch("auth.accounts._client", return_value=mock_client):
            success, msg = login("nobody@example.com", "pw")

        assert success is False
        assert "Invalid email or password" in msg

    def test_normalizes_email_before_lookup(self):
        hashed = _hashed("pw")
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [
            {"email": "a@b.com", "password_hash": hashed, "approved": True}
        ]

        with patch("auth.accounts._client", return_value=mock_client):
            login("A@B.COM", "pw")

        mock_client.table.return_value.select.return_value.eq.assert_called_with(
            "email", "a@b.com"
        )


class TestListAccounts:
    def test_returns_all_accounts(self):
        rows = [
            {"id": "1", "email": "a@b.com", "approved": True,
             "created_at": "2026-05-12T00:00:00Z", "approved_at": None}
        ]
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.order.return_value.execute.return_value.data = rows

        with patch("auth.accounts._client", return_value=mock_client):
            result = list_accounts()

        assert result == rows
        mock_client.table.return_value.select.return_value.order.assert_called_with("created_at")


class TestApproveAccount:
    def test_sets_approved_and_timestamp(self):
        mock_client = MagicMock()

        with patch("auth.accounts._client", return_value=mock_client):
            approve_account("some-uuid")

        updated = mock_client.table.return_value.update.call_args[0][0]
        assert updated["approved"] is True
        assert "approved_at" in updated
        mock_client.table.return_value.update.return_value.eq.assert_called_with(
            "id", "some-uuid"
        )
        from datetime import datetime
        datetime.fromisoformat(updated["approved_at"])  # raises ValueError if not valid ISO format
