"""
Lead CRUD service — pure database operations.

DESIGN RULE: Services contain business logic and DB queries.
             They do NOT know about HTTP, FastAPI, or the LLM.
             This makes them independently testable.

TENANT ISOLATION:
  Every query filters by tenant_id.
  This is enforced at the service layer — the API layer cannot bypass it
  because tenant_id always comes from the authenticated JWT, not user input.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.crm import Lead, LeadStatus
from app.schemas.lead import LeadCreate, LeadUpdate


class LeadService:
    def __init__(self, db: AsyncSession, tenant_id: str):
        self.db = db
        self.tenant_id = tenant_id

    # ── Create ────────────────────────────────────────────────────────────────

    async def create(self, data: LeadCreate, owner_id: str) -> Lead:
        lead = Lead(
            tenant_id=self.tenant_id,
            owner_id=owner_id,
            **data.model_dump(),
        )
        self.db.add(lead)
        await self.db.commit()
        await self.db.refresh(lead)
        return lead

    # ── Read ──────────────────────────────────────────────────────────────────

    async def get_by_id(self, lead_id: str) -> Optional[Lead]:
        result = await self.db.execute(
            select(Lead).where(
                Lead.id == lead_id,
                Lead.tenant_id == self.tenant_id,   # TENANT ISOLATION
            )
        )
        return result.scalar_one_or_none()

    async def list(
        self,
        page: int = 1,
        page_size: int = 20,
        status: Optional[LeadStatus] = None,
        owner_id: Optional[str] = None,
        search: Optional[str] = None,
    ) -> Tuple[List[Lead], int]:
        query = select(Lead).where(Lead.tenant_id == self.tenant_id)

        if status:
            query = query.where(Lead.status == status)
        if owner_id:
            query = query.where(Lead.owner_id == owner_id)
        if search:
            search_term = f"%{search}%"
            query = query.where(
                (Lead.name.ilike(search_term))
                | (Lead.company.ilike(search_term))
                | (Lead.email.ilike(search_term))
            )

        # Count total
        count_q = select(func.count()).select_from(query.subquery())
        total = (await self.db.execute(count_q)).scalar_one()

        # Paginate
        offset = (page - 1) * page_size
        query = query.order_by(Lead.created_at.desc()).offset(offset).limit(page_size)
        result = await self.db.execute(query)
        leads = result.scalars().all()

        return leads, total

    # ── Update ────────────────────────────────────────────────────────────────

    async def update(self, lead_id: str, data: LeadUpdate) -> Optional[Lead]:
        lead = await self.get_by_id(lead_id)
        if not lead:
            return None
        update_data = data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(lead, key, value)
        await self.db.commit()
        await self.db.refresh(lead)
        return lead

    # ── Delete ────────────────────────────────────────────────────────────────

    async def delete(self, lead_id: str) -> bool:
        """
        Hard delete. In production you may prefer soft delete (is_deleted flag).
        WHY confirmation required before calling this:
          Deletes are not retried. If the LLM calls this by mistake,
          data is gone. Application always asks for explicit confirmation.
        """
        lead = await self.get_by_id(lead_id)
        if not lead:
            return False
        await self.db.delete(lead)
        await self.db.commit()
        return True

    # ── Analytics helpers ─────────────────────────────────────────────────────

    async def count_by_status(self) -> dict:
        """Used by the AnalyticsAgent — deterministic SQL, not LLM math."""
        result = await self.db.execute(
            select(Lead.status, func.count(Lead.id))
            .where(Lead.tenant_id == self.tenant_id)
            .group_by(Lead.status)
        )
        return {row[0].value: row[1] for row in result.all()}
