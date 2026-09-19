"""Auth endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.security import create_access_token, verify_password
from app.database import get_db
from app.models.crm import User
from app.schemas.auth import LoginRequest, TokenResponse

router = APIRouter()


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, db: AsyncSession = Depends(get_db)):
    """
    Authenticate a user and return a JWT access token.
    The token payload contains: sub (user_id), tenant_id, role.
    These claims are used for RBAC and tenant isolation throughout the app.
    """
    result = await db.execute(select(User).where(User.email == request.email))
    user = result.scalar_one_or_none()

    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )

    token = create_access_token(
        data={"sub": user.id, "tenant_id": user.tenant_id, "role": user.role.value}
    )
    return TokenResponse(
        access_token=token,
        user_id=user.id,
        tenant_id=user.tenant_id,
        role=user.role.value,
        name=user.name,
    )
