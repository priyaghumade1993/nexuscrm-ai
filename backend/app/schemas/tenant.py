from datetime import datetime
from pydantic import BaseModel


class TenantCreate(BaseModel):
    name: str


class TenantOut(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    name: str
    is_active: bool
    created_at: datetime
