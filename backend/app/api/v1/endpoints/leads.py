"""Leads CRUD endpoints with full pagination, filtering, RBAC."""
import math
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import get_current_user, require_manager_or_above
from app.database import get_db
from app.models.crm import LeadStatus, User
from app.schemas.lead import LeadCreate, LeadListOut, LeadOut, LeadUpdate
from app.schemas.common import SuccessResponse
from app.services.lead_service import LeadService

router = APIRouter()


def _svc(db: AsyncSession, user: User) -> LeadService:
    return LeadService(db=db, tenant_id=user.tenant_id)


@router.get("/", response_model=LeadListOut)
async def list_leads(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[LeadStatus] = None,
    owner_id: Optional[str] = None,
    search: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List leads — auto-filtered to the current user's tenant."""
    leads, total = await _svc(db, current_user).list(
        page=page, page_size=page_size, status=status,
        owner_id=owner_id, search=search,
    )
    return LeadListOut(
        items=leads, total=total, page=page, page_size=page_size,
        pages=math.ceil(total / page_size) if total else 0,
    )


@router.post("/", response_model=LeadOut, status_code=status.HTTP_201_CREATED)
async def create_lead(
    data: LeadCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _svc(db, current_user).create(data, owner_id=current_user.id)


@router.get("/{lead_id}", response_model=LeadOut)
async def get_lead(
    lead_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    lead = await _svc(db, current_user).get_by_id(lead_id)
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    return lead


@router.patch("/{lead_id}", response_model=LeadOut)
async def update_lead(
    lead_id: str,
    data: LeadUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    lead = await _svc(db, current_user).update(lead_id, data)
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    return lead


@router.delete("/{lead_id}", response_model=SuccessResponse)
async def delete_lead(
    lead_id: str,
    confirm: bool = Query(False, description="Set to true to confirm deletion"),
    current_user: User = Depends(require_manager_or_above()),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a lead.
    WHY explicit confirm flag:
      Destructive operations require intentional confirmation.
      The LLM cannot set confirm=True on its own — the user must explicitly
      pass it via the API or the chat confirm_destructive=True field.
    """
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Deletion requires confirm=true query parameter",
        )
    deleted = await _svc(db, current_user).delete(lead_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Lead not found")
    return SuccessResponse(message="Lead deleted", id=lead_id)
