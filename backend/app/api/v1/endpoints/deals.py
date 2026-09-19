import math
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import get_current_user, require_manager_or_above
from app.database import get_db
from app.models.crm import DealStage, User
from app.schemas.deal import DealCreate, DealListOut, DealOut, DealUpdate
from app.schemas.common import SuccessResponse
from app.services.deal_service import DealService

router = APIRouter()


def _svc(db: AsyncSession, user: User) -> DealService:
    return DealService(db=db, tenant_id=user.tenant_id)


@router.get("/", response_model=DealListOut)
async def list_deals(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    stage: Optional[DealStage] = None,
    owner_id: Optional[str] = None,
    search: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    deals, total = await _svc(db, current_user).list(
        page=page, page_size=page_size, stage=stage, owner_id=owner_id, search=search
    )
    return DealListOut(
        items=deals, total=total, page=page, page_size=page_size,
        pages=math.ceil(total / page_size) if total else 0,
    )


@router.post("/", response_model=DealOut, status_code=status.HTTP_201_CREATED)
async def create_deal(
    data: DealCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _svc(db, current_user).create(data, owner_id=current_user.id)


@router.get("/{deal_id}", response_model=DealOut)
async def get_deal(
    deal_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    deal = await _svc(db, current_user).get_by_id(deal_id)
    if not deal:
        raise HTTPException(status_code=404, detail="Deal not found")
    return deal


@router.patch("/{deal_id}", response_model=DealOut)
async def update_deal(
    deal_id: str,
    data: DealUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    deal = await _svc(db, current_user).update(deal_id, data)
    if not deal:
        raise HTTPException(status_code=404, detail="Deal not found")
    return deal


@router.delete("/{deal_id}", response_model=SuccessResponse)
async def delete_deal(
    deal_id: str,
    confirm: bool = Query(False),
    current_user: User = Depends(require_manager_or_above()),
    db: AsyncSession = Depends(get_db),
):
    if not confirm:
        raise HTTPException(status_code=400, detail="Deletion requires confirm=true")
    deleted = await _svc(db, current_user).delete(deal_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Deal not found")
    return SuccessResponse(message="Deal deleted", id=deal_id)
