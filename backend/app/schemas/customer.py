from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class CustomerCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    email: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    industry: Optional[str] = None
    website: Optional[str] = None
    address: Optional[str] = None


class CustomerUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    industry: Optional[str] = None
    website: Optional[str] = None
    address: Optional[str] = None


class CustomerOut(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    tenant_id: str
    name: str
    email: Optional[str]
    phone: Optional[str]
    company: Optional[str]
    industry: Optional[str]
    website: Optional[str]
    address: Optional[str]
    created_at: datetime
    updated_at: datetime


class CustomerListOut(BaseModel):
    items: List[CustomerOut]
    total: int
    page: int
    page_size: int
    pages: int
