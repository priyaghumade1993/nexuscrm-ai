from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from app.models.crm import LeadSource, LeadStatus


class LeadCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    email: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    source: LeadSource = LeadSource.other
    status: LeadStatus = LeadStatus.new
    score: int = Field(default=0, ge=0, le=100)
    notes: Optional[str] = None


class LeadUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    email: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    source: Optional[LeadSource] = None
    status: Optional[LeadStatus] = None
    score: Optional[int] = Field(None, ge=0, le=100)
    notes: Optional[str] = None


class LeadOut(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    tenant_id: str
    owner_id: Optional[str]
    name: str
    email: Optional[str]
    phone: Optional[str]
    company: Optional[str]
    source: LeadSource
    status: LeadStatus
    score: int
    notes: Optional[str]
    created_at: datetime
    updated_at: datetime


class LeadListOut(BaseModel):
    items: List[LeadOut]
    total: int
    page: int
    page_size: int
    pages: int
