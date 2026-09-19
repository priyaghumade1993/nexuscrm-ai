import math
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import get_current_user, require_manager_or_above
from app.database import get_db
from app.models.crm import User
from app.schemas.customer import CustomerCreate, CustomerListOut, CustomerOut, CustomerUpdate
from app.schemas.common import SuccessResponse
from app.services.customer_service import CustomerService

router = APIRouter()


def _svc(db: AsyncSession, user: User) -> CustomerService:
    return CustomerService(db=db, tenant_id=user.tenant_id)


@router.get("/", response_model=CustomerListOut)
async def list_customers(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    industry: Optional[str] = None,
    search: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    customers, total = await _svc(db, current_user).list(
        page=page, page_size=page_size, industry=industry, search=search
    )
    return CustomerListOut(
        items=customers, total=total, page=page, page_size=page_size,
        pages=math.ceil(total / page_size) if total else 0,
    )


@router.post("/", response_model=CustomerOut, status_code=status.HTTP_201_CREATED)
async def create_customer(
    data: CustomerCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _svc(db, current_user).create(data)


@router.get("/{customer_id}", response_model=CustomerOut)
async def get_customer(
    customer_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    customer = await _svc(db, current_user).get_by_id(customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    return customer


@router.patch("/{customer_id}", response_model=CustomerOut)
async def update_customer(
    customer_id: str,
    data: CustomerUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    customer = await _svc(db, current_user).update(customer_id, data)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    return customer


@router.delete("/{customer_id}", response_model=SuccessResponse)
async def delete_customer(
    customer_id: str,
    confirm: bool = Query(False),
    current_user: User = Depends(require_manager_or_above()),
    db: AsyncSession = Depends(get_db),
):
    if not confirm:
        raise HTTPException(status_code=400, detail="Deletion requires confirm=true")
    deleted = await _svc(db, current_user).delete(customer_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Customer not found")
    return SuccessResponse(message="Customer deleted", id=customer_id)
