"""
Authentication & security utilities.

WHY passlib + bcrypt:
  bcrypt is the gold standard for password hashing — computationally expensive
  by design (making brute-force attacks slow), with built-in salting.
  NEVER use MD5, SHA256, or plain SHA for passwords.

WHY JWT:
  Stateless authentication — the server doesn't need a session store.
  The token carries the user's id, tenant_id, and role — all claims the
  application needs without an extra DB lookup per request.

IMPORTANT SECURITY RULE:
  Authorization (RBAC, tenant isolation) is enforced by APPLICATION CODE.
  The LLM does NOT decide whether a user is authorized.
  Every tool validates the current user's role and tenant_id before executing.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.config import settings

# ── Password hashing ──────────────────────────────────────────────────────────
# bcrypt with deprecated="auto" automatically upgrades weak hashes on next login.
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    """Return bcrypt hash of the plain-text password."""
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Verify plain-text password against stored bcrypt hash."""
    return pwd_context.verify(plain, hashed)


# ── JWT ───────────────────────────────────────────────────────────────────────

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a signed JWT access token.

    The 'sub' claim is the user_id.
    We also embed tenant_id and role so downstream tools can enforce isolation
    without an extra DB lookup.
    """
    payload = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    payload["exp"] = expire
    payload["iat"] = datetime.now(timezone.utc)
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate a JWT.
    Raises JWTError if invalid or expired — caught by the auth dependency.
    """
    return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
