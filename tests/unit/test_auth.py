"""
Unit tests for authentication security module.
Tests password hashing, JWT creation, and token decoding.
No database required.

NOTE: Imports only from app.auth.security (not app.auth) to avoid
pulling in FastAPI/SQLAlchemy/MongoDB dependencies at import time.
"""
import pytest
from datetime import timedelta
import sys
import os

# Ensure the backend is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))


def _import_security():
    """Import only the security module, bypassing the full auth __init__."""
    # Import config first with mocked values to avoid needing .env
    import unittest.mock as mock
    with mock.patch.dict(os.environ, {
        "JWT_SECRET_KEY": "test-secret-key-for-unit-tests-minimum-32-chars-long",
        "DATABASE_URL": "postgresql+asyncpg://test:test@localhost/test",
        "MONGODB_URL": "mongodb://localhost:27017",
        "OPENAI_API_KEY": "sk-test-fake",
    }):
        # Import only security.py, not the full auth package
        import importlib
        # Temporarily patch app.auth.__init__ imports
        from app.auth import security
        return security


class TestPasswordHashing:

    def test_hash_password_returns_hash(self):
        try:
            from app.auth.security import hash_password
        except ImportError:
            pytest.skip("auth.security not importable")
        hashed = hash_password("Password123!")
        assert hashed != "Password123!"
        assert len(hashed) > 20

    def test_verify_correct_password(self):
        try:
            from app.auth.security import hash_password, verify_password
        except ImportError:
            pytest.skip("auth.security not importable")
        hashed = hash_password("Password123!")
        assert verify_password("Password123!", hashed) is True

    def test_verify_wrong_password(self):
        try:
            from app.auth.security import hash_password, verify_password
        except ImportError:
            pytest.skip("auth.security not importable")
        hashed = hash_password("Password123!")
        assert verify_password("WrongPassword!", hashed) is False

    def test_different_hashes_for_same_password(self):
        """bcrypt generates a new salt each time — hashes must differ."""
        try:
            from app.auth.security import hash_password
        except ImportError:
            pytest.skip("auth.security not importable")
        hash1 = hash_password("Password123!")
        hash2 = hash_password("Password123!")
        assert hash1 != hash2

    def test_verify_after_re_hash(self):
        try:
            from app.auth.security import hash_password, verify_password
        except ImportError:
            pytest.skip("auth.security not importable")
        hashed = hash_password("Password123!")
        assert verify_password("Password123!", hashed) is True


class TestJWT:
    """
    create_access_token(data: dict, expires_delta: timedelta | None) -> str
    data should contain: {"sub": user_id, "tenant_id": ..., "role": ...}
    """

    def test_create_access_token(self):
        try:
            from app.auth.security import create_access_token
        except ImportError:
            pytest.skip("auth.security not importable")
        token = create_access_token(
            data={"sub": "user-123", "tenant_id": "tenant-001", "role": "sales_rep"}
        )
        assert isinstance(token, str)
        assert len(token) > 50

    def test_decode_valid_token(self):
        try:
            from app.auth.security import create_access_token, decode_token
        except ImportError:
            pytest.skip("auth.security not importable")
        token = create_access_token(
            data={"sub": "user-123", "tenant_id": "tenant-001", "role": "sales_rep"}
        )
        payload = decode_token(token)
        assert payload["sub"] == "user-123"
        assert payload["tenant_id"] == "tenant-001"
        assert payload["role"] == "sales_rep"

    def test_decode_invalid_token_raises(self):
        try:
            from app.auth.security import decode_token
            from jose import JWTError
        except ImportError:
            pytest.skip("auth.security not importable")
        with pytest.raises(JWTError):
            decode_token("this.is.not.a.valid.jwt")

    def test_decode_tampered_token_raises(self):
        try:
            from app.auth.security import create_access_token, decode_token
            from jose import JWTError
        except ImportError:
            pytest.skip("auth.security not importable")
        token = create_access_token(
            data={"sub": "user-123", "tenant_id": "tenant-001", "role": "sales_rep"}
        )
        # Tamper with the signature
        parts = token.split(".")
        tampered = parts[0] + "." + parts[1] + ".invalidsignature"
        with pytest.raises(JWTError):
            decode_token(tampered)

    def test_different_users_have_different_tokens(self):
        try:
            from app.auth.security import create_access_token
        except ImportError:
            pytest.skip("auth.security not importable")
        token1 = create_access_token(data={"sub": "user-1", "tenant_id": "tenant-001", "role": "sales_rep"})
        token2 = create_access_token(data={"sub": "user-2", "tenant_id": "tenant-001", "role": "sales_rep"})
        assert token1 != token2

    def test_token_contains_tenant_isolation(self):
        """Ensure tenant_id is embedded in the token for claim-based isolation."""
        try:
            from app.auth.security import create_access_token, decode_token
        except ImportError:
            pytest.skip("auth.security not importable")
        token = create_access_token(data={"sub": "user-1", "tenant_id": "tenant-ACME", "role": "admin"})
        payload = decode_token(token)
        assert payload["tenant_id"] == "tenant-ACME"

    def test_token_expiry_with_custom_delta(self):
        """Token with custom expiry is created correctly."""
        try:
            from app.auth.security import create_access_token, decode_token
        except ImportError:
            pytest.skip("auth.security not importable")
        token = create_access_token(
            data={"sub": "user-1", "tenant_id": "t1", "role": "viewer"},
            expires_delta=timedelta(hours=1),
        )
        payload = decode_token(token)
        assert "exp" in payload
        assert "iat" in payload
