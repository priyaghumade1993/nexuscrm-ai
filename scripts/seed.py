"""
NexusCRM Database Seed Script.

Creates realistic demo data for development and portfolio showcasing.

USAGE:
  cd backend
  python ../scripts/seed.py

WHAT IT CREATES:
  - 1 Tenant (Geekhub Tech Solutions)
  - 3 Users (admin, manager, sales_rep)
  - 10 Leads with varied statuses
  - 5 Customers (enterprise accounts)
  - 8 Deals across all pipeline stages
  - 5 Activities
  - 3 Products

PASSWORD FOR ALL DEMO USERS: Password123!
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from decimal import Decimal
from datetime import date, timedelta
import uuid


async def seed():
    from app.config.settings import get_settings
    from app.database.base import engine, Base
    from app.database.base import AsyncSessionLocal
    from app.auth.security import hash_password
    from app.models.crm import (
        Tenant, User, Lead, Customer, Deal, Activity, Product,
        UserRole, LeadStatus, LeadSource, DealStage,
        ActivityType, ActivityStatus,
    )
    from sqlalchemy import select

    settings = get_settings()
    print(f"Seeding database: {settings.database_url[:50]}...")

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Tables created/verified.")

    async with AsyncSessionLocal() as session:

        # ── Tenant ────────────────────────────────────────────────────────────
        tenant_id = "tenant-geekhub-demo"
        existing = await session.get(Tenant, tenant_id)
        if existing:
            print("Seed data already exists. Skipping.")
            return

        tenant = Tenant(
            id=tenant_id,
            name="Geekhub Tech Solutions",
            domain="geekhub.io",
            plan="enterprise",
            is_active=True,
        )
        session.add(tenant)
        await session.flush()
        print(f"  ✓ Tenant: {tenant.name}")

        # ── Users ─────────────────────────────────────────────────────────────
        pw_hash = hash_password("Password123!")

        admin = User(
            id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            email="admin@geekhub.io",
            hashed_password=pw_hash,
            full_name="Admin User",
            role=UserRole.admin,
            is_active=True,
        )
        manager = User(
            id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            email="manager@geekhub.io",
            hashed_password=pw_hash,
            full_name="Arjun Sharma (Manager)",
            role=UserRole.manager,
            is_active=True,
        )
        sales_rep = User(
            id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            email="priya@geekhub.io",
            hashed_password=pw_hash,
            full_name="Priya Tsurkar (Sales Rep)",
            role=UserRole.sales_rep,
            is_active=True,
        )
        viewer = User(
            id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            email="viewer@geekhub.io",
            hashed_password=pw_hash,
            full_name="Viewer User",
            role=UserRole.viewer,
            is_active=True,
        )
        session.add_all([admin, manager, sales_rep, viewer])
        await session.flush()
        print(f"  ✓ Users: admin, manager, sales_rep, viewer (all password: Password123!)")

        # ── Products ──────────────────────────────────────────────────────────
        products = [
            Product(id=str(uuid.uuid4()), tenant_id=tenant_id,
                    name="NexusCRM Starter", sku="NX-STARTER",
                    price=Decimal("29.00"), category="saas"),
            Product(id=str(uuid.uuid4()), tenant_id=tenant_id,
                    name="NexusCRM Professional", sku="NX-PRO",
                    price=Decimal("79.00"), category="saas"),
            Product(id=str(uuid.uuid4()), tenant_id=tenant_id,
                    name="NexusCRM Enterprise", sku="NX-ENT",
                    price=Decimal("199.00"), category="saas"),
            Product(id=str(uuid.uuid4()), tenant_id=tenant_id,
                    name="AI Add-on (Professional)", sku="NX-AI-PRO",
                    price=Decimal("20.00"), category="addon"),
            Product(id=str(uuid.uuid4()), tenant_id=tenant_id,
                    name="AI Add-on (Enterprise)", sku="NX-AI-ENT",
                    price=Decimal("35.00"), category="addon"),
        ]
        session.add_all(products)
        await session.flush()
        print(f"  ✓ Products: {len(products)} created")

        # ── Leads ─────────────────────────────────────────────────────────────
        leads_data = [
            ("Rahul Mehta", "rahul.mehta@infosys.com", "Infosys", LeadStatus.qualified, LeadSource.linkedin, 85, sales_rep.id),
            ("Ananya Krishnan", "ananya@wipro.com", "Wipro Technologies", LeadStatus.new, LeadSource.website, 60, sales_rep.id),
            ("Vikram Singh", "vikram.singh@tcs.com", "TCS", LeadStatus.hot, LeadSource.referral, 92, sales_rep.id),
            ("Meera Patel", "meera@hcl.com", "HCL Technologies", LeadStatus.qualified, LeadSource.email_campaign, 70, sales_rep.id),
            ("Arjun Nair", "arjun.n@techm.com", "Tech Mahindra", LeadStatus.new, LeadSource.cold_call, 45, manager.id),
            ("Sunita Rao", "sunita@mindtree.com", "Mindtree", LeadStatus.contacted, LeadSource.conference, 55, sales_rep.id),
            ("Deepak Kumar", "deepak@mphasis.com", "Mphasis", LeadStatus.warm, LeadSource.website, 65, sales_rep.id),
            ("Kavya Reddy", "kavya@hexaware.com", "Hexaware", LeadStatus.new, LeadSource.linkedin, 40, sales_rep.id),
            ("Nikhil Sharma", "nikhil@cyient.com", "Cyient", LeadStatus.cold, LeadSource.email_campaign, 20, manager.id),
            ("Pooja Iyer", "pooja@birlasoft.com", "Birlasoft", LeadStatus.qualified, LeadSource.referral, 78, sales_rep.id),
        ]
        leads = []
        for name, email, company, status, source, score, owner_id in leads_data:
            lead = Lead(
                id=str(uuid.uuid4()),
                tenant_id=tenant_id,
                owner_id=owner_id,
                name=name,
                email=email,
                company=company,
                status=status,
                source=source,
                lead_score=score,
            )
            leads.append(lead)
        session.add_all(leads)
        await session.flush()
        print(f"  ✓ Leads: {len(leads)} created")

        # ── Customers ─────────────────────────────────────────────────────────
        customers_data = [
            ("TCS Enterprise", "tcs.com", "Information Technology", "India", 614000),
            ("Infosys Global", "infosys.com", "Information Technology", "India", 335000),
            ("Wipro Technologies", "wipro.com", "Information Technology", "India", 250000),
            ("HCL Technologies", "hcltech.com", "Information Technology", "India", 225000),
            ("Tech Mahindra", "techmahindra.com", "Telecom IT", "India", 158000),
        ]
        customers = []
        for name, domain, industry, country, employees in customers_data:
            customer = Customer(
                id=str(uuid.uuid4()),
                tenant_id=tenant_id,
                company_name=name,
                domain=domain,
                industry=industry,
                country=country,
                employee_count=employees,
                account_manager_id=manager.id,
            )
            customers.append(customer)
        session.add_all(customers)
        await session.flush()
        print(f"  ✓ Customers: {len(customers)} created")

        # ── Deals ─────────────────────────────────────────────────────────────
        today = date.today()
        deals_data = [
            ("TCS Enterprise CRM License", customers[0].id, DealStage.negotiation,
             Decimal("450000.00"), today + timedelta(days=15), sales_rep.id),
            ("Infosys AI Add-on Bundle", customers[1].id, DealStage.proposal_sent,
             Decimal("280000.00"), today + timedelta(days=30), sales_rep.id),
            ("Wipro Starter Rollout", customers[2].id, DealStage.demo_completed,
             Decimal("95000.00"), today + timedelta(days=45), sales_rep.id),
            ("HCL Professional Plan", customers[3].id, DealStage.demo_scheduled,
             Decimal("180000.00"), today + timedelta(days=60), manager.id),
            ("Tech Mahindra Enterprise", customers[4].id, DealStage.qualified,
             Decimal("320000.00"), today + timedelta(days=75), sales_rep.id),
            ("Infosys Additional Seats", customers[1].id, DealStage.closed_won,
             Decimal("120000.00"), today - timedelta(days=10), sales_rep.id),
            ("Small Startup - Starter", None, DealStage.prospect,
             Decimal("12000.00"), today + timedelta(days=90), sales_rep.id),
            ("Old Deal - Lost to HubSpot", customers[2].id, DealStage.closed_lost,
             Decimal("75000.00"), today - timedelta(days=30), manager.id),
        ]
        deals = []
        for name, customer_id, stage, amount, close_date, owner_id in deals_data:
            deal = Deal(
                id=str(uuid.uuid4()),
                tenant_id=tenant_id,
                name=name,
                customer_id=customer_id,
                owner_id=owner_id,
                stage=stage,
                amount=amount,
                expected_close_date=close_date,
                probability=_stage_probability(stage),
            )
            deals.append(deal)
        session.add_all(deals)
        await session.flush()
        total_pipeline = sum(d.amount for d in deals if d.stage not in
                             [DealStage.closed_won, DealStage.closed_lost])
        print(f"  ✓ Deals: {len(deals)} created | Active pipeline: ${total_pipeline:,.2f}")

        await session.commit()

    print("\n✅ Seed complete!")
    print("\nDemo credentials:")
    print("  Admin:    admin@geekhub.io     / Password123!")
    print("  Manager:  manager@geekhub.io   / Password123!")
    print("  Sales:    priya@geekhub.io     / Password123!")
    print("  Viewer:   viewer@geekhub.io    / Password123!")
    print(f"\nTenant ID: {tenant_id}")


def _stage_probability(stage: "DealStage") -> int:
    from app.models.crm import DealStage
    return {
        DealStage.prospect: 10,
        DealStage.qualified: 20,
        DealStage.demo_scheduled: 30,
        DealStage.demo_completed: 40,
        DealStage.proposal_sent: 60,
        DealStage.negotiation: 75,
        DealStage.closed_won: 100,
        DealStage.closed_lost: 0,
    }.get(stage, 50)


if __name__ == "__main__":
    asyncio.run(seed())
