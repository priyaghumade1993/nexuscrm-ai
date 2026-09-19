"""
FastAPI auth dependencies — current user extraction and RBAC enforcement.

DESIGN PATTERN:
  Every protected endpoint declares `current_user: User = Depends(get_current_user)`.
  Role-specific endpoints further declare `_: User = Depends(require_role("admin"))`.

WHY dependency injection (not middleware):
  Endpoint-specific auth makes the required role explicit in the function signature.
  It is testable — you can inject a mock user in tests.
  It avoids the complexity of thread-local state.
"""
from __future__ import annotations

from typing import List

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.security import decode_token
from app.database import get_db
from app.models.crm import User, UserRole

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Validate JWT and return the authenticated User ORM object.
    Raises 401 on invalid token, 403 on inactive account.
    """
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_token(token)
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exc
    except JWTError:
        raise credentials_exc

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise credentials_exc
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user")
    return user


def require_role(*roles: UserRole):
    """
    Dependency factory — returns a dependency that enforces the required role(s).

    Usage:
        @router.delete("/{id}")
        async def delete_lead(
            _: User = Depends(require_role(UserRole.admin, UserRole.manager)),
            ...
        ):
    """
    async def check_role(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {[r.value for r in roles]}",
            )
        return current_user
    return check_role


# Convenience shortcuts
require_admin = require_role(UserRole.admin)
require_manager_or_above = require_role(UserRole.admin, UserRole.manager)
require_sales_or_above = require_role(UserRole.admin, UserRole.manager, UserRole.sales_rep)
