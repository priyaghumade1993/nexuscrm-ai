"""Shared Pydantic schemas."""
from __future__ import annotations

from typing import Generic, List, Optional, TypeVar

from pydantic import BaseModel, ConfigDict

T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    """
    Standard paginated list response.
    WHY: consistent shape for all list endpoints — Streamlit and JS client
         can use the same parsing logic regardless of resource type.
    """
    items: List[T]
    total: int
    page: int
    page_size: int
    pages: int


class ErrorResponse(BaseModel):
    """Structured error response — never expose raw stack traces."""
    detail: str
    error_code: Optional[str] = None
    field: Optional[str] = None


class SuccessResponse(BaseModel):
    message: str
    id: Optional[str] = None
