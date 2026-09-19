from __future__ import annotations

from typing import List, Optional, Tuple

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.crm import Customer, Interaction
from app.schemas.customer import CustomerCreate, CustomerUpdate


class CustomerService:
    def __init__(self, db: AsyncSession, tenant_id: str):
        self.db = db
        self.tenant_id = tenant_id

    async def create(self, data: CustomerCreate) -> Customer:
        customer = Customer(tenant_id=self.tenant_id, **data.model_dump())
        self.db.add(customer)
        await self.db.commit()
        await self.db.refresh(customer)
        return customer

    async def get_by_id(self, customer_id: str) -> Optional[Customer]:
        result = await self.db.execute(
            select(Customer).where(
                Customer.id == customer_id,
                Customer.tenant_id == self.tenant_id,   # TENANT ISOLATION
            )
        )
        return result.scalar_one_or_none()

    async def list(
        self,
        page: int = 1,
        page_size: int = 20,
        industry: Optional[str] = None,
        search: Optional[str] = None,
    ) -> Tuple[List[Customer], int]:
        query = select(Customer).where(Customer.tenant_id == self.tenant_id)
        if industry:
            query = query.where(Customer.industry.ilike(f"%{industry}%"))
        if search:
            term = f"%{search}%"
            query = query.where(
                (Customer.name.ilike(term))
                | (Customer.company.ilike(term))
                | (Customer.email.ilike(term))
            )
        count_q = select(func.count()).select_from(query.subquery())
        total = (await self.db.execute(count_q)).scalar_one()
        offset = (page - 1) * page_size
        query = query.order_by(Customer.created_at.desc()).offset(offset).limit(page_size)
        result = await self.db.execute(query)
        return result.scalars().all(), total

    async def update(self, customer_id: str, data: CustomerUpdate) -> Optional[Customer]:
        customer = await self.get_by_id(customer_id)
        if not customer:
            return None
        for key, value in data.model_dump(exclude_unset=True).items():
            setattr(customer, key, value)
        await self.db.commit()
        await self.db.refresh(customer)
        return customer

    async def delete(self, customer_id: str) -> bool:
        customer = await self.get_by_id(customer_id)
        if not customer:
            return False
        await self.db.delete(customer)
        await self.db.commit()
        return True

    async def get_interactions(self, customer_id: str) -> List[Interaction]:
        """Get all interaction history for a customer (same tenant)."""
        result = await self.db.execute(
            select(Interaction)
            .where(
                Interaction.customer_id == customer_id,
                Interaction.tenant_id == self.tenant_id,
            )
            .order_by(Interaction.timestamp.desc())
            .limit(50)
        )
        return result.scalars().all()
