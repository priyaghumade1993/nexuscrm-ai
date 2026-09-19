from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from app.models.crm import DealStage


class DealCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=300)
    customer_id: Optional[str] = None
    amount: Optional[float] = Field(None, ge=0)
    stage: DealStage = DealStage.prospecting
    probability: int = Field(default=0, ge=0, le=100)
    expected_close_date: Optional[datetime] = None
    description: Optional[str] = None


class DealUpdate(BaseModel):
    title: Optional[str] = None
    customer_id: Optional[str] = None
    amount: Optional[float] = Field(None, ge=0)
    stage: Optional[DealStage] = None
    probability: Optional[int] = Field(None, ge=0, le=100)
    expected_close_date: Optional[datetime] = None
    description: Optional[str] = None


class DealOut(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    tenant_id: str
    owner_id: Optional[str]
    customer_id: Optional[str]
    title: str
    amount: Optional[float]
    stage: DealStage
    probability: int
    expected_close_date: Optional[datetime]
    description: Optional[str]
    created_at: datetime
    updated_at: datetime


class DealListOut(BaseModel):
    items: List[DealOut]
    total: int
    page: int
    page_size: int
    pages: int
