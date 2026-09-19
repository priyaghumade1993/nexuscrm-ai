from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr
from app.models.crm import UserRole


class UserCreate(BaseModel):
    name: str
    email: str
    password: str
    role: UserRole = UserRole.sales_rep


class UserUpdate(BaseModel):
    name: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


class UserOut(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    tenant_id: str
    name: str
    email: str
    role: UserRole
    is_active: bool
    created_at: datetime
