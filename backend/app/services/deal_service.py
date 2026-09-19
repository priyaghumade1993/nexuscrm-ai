from __future__ import annotations

from typing import List, Optional, Tuple

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.crm import Deal, DealStage
from app.schemas.deal import DealCreate, DealUpdate


class DealService:
    def __init__(self, db: AsyncSession, tenant_id: str):
        self.db = db
        self.tenant_id = tenant_id

    async def create(self, data: DealCreate, owner_id: str) -> Deal:
        deal = Deal(tenant_id=self.tenant_id, owner_id=owner_id, **data.model_dump())
        self.db.add(deal)
        await self.db.commit()
        await self.db.refresh(deal)
        return deal

    async def get_by_id(self, deal_id: str) -> Optional[Deal]:
        result = await self.db.execute(
            select(Deal).where(Deal.id == deal_id, Deal.tenant_id == self.tenant_id)
        )
        return result.scalar_one_or_none()

    async def list(
        self,
        page: int = 1,
        page_size: int = 20,
        stage: Optional[DealStage] = None,
        owner_id: Optional[str] = None,
        search: Optional[str] = None,
    ) -> Tuple[List[Deal], int]:
        query = select(Deal).where(Deal.tenant_id == self.tenant_id)
        if stage:
            query = query.where(Deal.stage == stage)
        if owner_id:
            query = query.where(Deal.owner_id == owner_id)
        if search:
            query = query.where(Deal.title.ilike(f"%{search}%"))
        count_q = select(func.count()).select_from(query.subquery())
        total = (await self.db.execute(count_q)).scalar_one()
        offset = (page - 1) * page_size
        query = query.order_by(Deal.created_at.desc()).offset(offset).limit(page_size)
        result = await self.db.execute(query)
        return result.scalars().all(), total

    async def update(self, deal_id: str, data: DealUpdate) -> Optional[Deal]:
        deal = await self.get_by_id(deal_id)
        if not deal:
            return None
        for key, value in data.model_dump(exclude_unset=True).items():
            setattr(deal, key, value)
        await self.db.commit()
        await self.db.refresh(deal)
        return deal

    async def delete(self, deal_id: str) -> bool:
        deal = await self.get_by_id(deal_id)
        if not deal:
            return False
        await self.db.delete(deal)
        await self.db.commit()
        return True

    async def calculate_pipeline_value(
        self, stage: Optional[DealStage] = None, open_only: bool = True
    ) -> dict:
        """
        Deterministic pipeline calculation — done in SQL, NOT by the LLM.
        WHY: LLMs can hallucinate numbers. Money calculations must be exact.
        """
        query = select(
            func.sum(Deal.amount).label("total"),
            func.count(Deal.id).label("count"),
            func.avg(Deal.amount).label("avg"),
        ).where(Deal.tenant_id == self.tenant_id)

        if open_only:
            query = query.where(
                Deal.stage.not_in([DealStage.closed_won, DealStage.closed_lost])
            )
        if stage:
            query = query.where(Deal.stage == stage)

        result = await self.db.execute(query)
        row = result.one()
        return {
            "total_pipeline_value": float(row.total or 0),
            "deal_count": row.count or 0,
            "average_deal_size": float(row.avg or 0),
        }

    async def get_open_deals_closing_this_month(self) -> List[Deal]:
        """Used by agents for 'what closes this month' queries."""
        from datetime import datetime, timezone
        import calendar
        now = datetime.now(timezone.utc)
        last_day = calendar.monthrange(now.year, now.month)[1]
        month_end = datetime(now.year, now.month, last_day, 23, 59, 59, tzinfo=timezone.utc)
        result = await self.db.execute(
            select(Deal)
            .where(
                Deal.tenant_id == self.tenant_id,
                Deal.expected_close_date <= month_end,
                Deal.expected_close_date >= now,
                Deal.stage.not_in([DealStage.closed_won, DealStage.closed_lost]),
            )
            .order_by(Deal.amount.desc())
        )
        return result.scalars().all()
