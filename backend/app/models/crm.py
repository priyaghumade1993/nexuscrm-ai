"""
NexusCRM AI — SQLAlchemy ORM models (PostgreSQL, SQLAlchemy 2.0 style).

TENANT ISOLATION DESIGN:
  Every tenant-scoped table has a tenant_id column.
  The application layer ALWAYS filters by tenant_id on every query.
  This is "application-level multi-tenancy" — the simplest and most portable
  approach. Row-level security (Postgres RLS) could be added later for
  defence-in-depth.

  NEVER let the LLM construct or bypass tenant filters.
  The tools enforce tenant_id from the authenticated user's JWT claims.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_uuid() -> str:
    return str(uuid.uuid4())


# ── Enums ─────────────────────────────────────────────────────────────────────

class UserRole(str, enum.Enum):
    admin = "admin"
    manager = "manager"
    sales_rep = "sales_rep"
    viewer = "viewer"


class LeadStatus(str, enum.Enum):
    new = "new"
    contacted = "contacted"
    qualified = "qualified"
    unqualified = "unqualified"
    converted = "converted"


class LeadSource(str, enum.Enum):
    website = "website"
    referral = "referral"
    cold_call = "cold_call"
    email_campaign = "email_campaign"
    social_media = "social_media"
    trade_show = "trade_show"
    partner = "partner"
    other = "other"


class DealStage(str, enum.Enum):
    prospecting = "prospecting"
    qualification = "qualification"
    proposal = "proposal"
    negotiation = "negotiation"
    closed_won = "closed_won"
    closed_lost = "closed_lost"


class ActivityType(str, enum.Enum):
    call = "call"
    email = "email"
    meeting = "meeting"
    demo = "demo"
    follow_up = "follow_up"
    task = "task"


class ActivityStatus(str, enum.Enum):
    pending = "pending"
    completed = "completed"
    cancelled = "cancelled"


class InteractionChannel(str, enum.Enum):
    email = "email"
    phone = "phone"
    chat = "chat"
    in_person = "in_person"
    video = "video"


# ── Tenant ────────────────────────────────────────────────────────────────────

class Tenant(Base):
    """
    Top-level isolation boundary. Every data-bearing table references this.
    WHY: SaaS CRM must never leak data between companies (tenants).
    """
    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=new_uuid
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )

    # Relationships
    users: Mapped[list[User]] = relationship("User", back_populates="tenant")
    leads: Mapped[list[Lead]] = relationship("Lead", back_populates="tenant")
    customers: Mapped[list[Customer]] = relationship("Customer", back_populates="tenant")
    deals: Mapped[list[Deal]] = relationship("Deal", back_populates="tenant")


# ── User ──────────────────────────────────────────────────────────────────────

class User(Base):
    """
    CRM user. Always belongs to exactly one tenant.
    role drives RBAC: admin > manager > sales_rep > viewer.
    """
    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("tenant_id", "email", name="uq_user_tenant_email"),
        Index("ix_users_tenant_id", "tenant_id"),
        Index("ix_users_email", "email"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole), nullable=False, default=UserRole.sales_rep
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )

    # Relationships
    tenant: Mapped[Tenant] = relationship("Tenant", back_populates="users")
    leads: Mapped[list[Lead]] = relationship("Lead", back_populates="owner")
    deals: Mapped[list[Deal]] = relationship("Deal", back_populates="owner")
    activities: Mapped[list[Activity]] = relationship("Activity", back_populates="user")
    notes: Mapped[list[Note]] = relationship("Note", back_populates="user")


# ── Lead ──────────────────────────────────────────────────────────────────────

class Lead(Base):
    """
    Sales lead — a potential customer not yet converted.
    TENANT ISOLATION: always filter by tenant_id.
    score: 0–100, set by application logic (not LLM).
    """
    __tablename__ = "leads"
    __table_args__ = (
        Index("ix_leads_tenant_id", "tenant_id"),
        Index("ix_leads_owner_id", "owner_id"),
        Index("ix_leads_status", "status"),
        Index("ix_leads_tenant_status", "tenant_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    owner_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    email: Mapped[str] = mapped_column(String(320), nullable=True)
    phone: Mapped[str] = mapped_column(String(50), nullable=True)
    company: Mapped[str] = mapped_column(String(200), nullable=True)
    source: Mapped[LeadSource] = mapped_column(
        Enum(LeadSource), nullable=False, default=LeadSource.other
    )
    status: Mapped[LeadStatus] = mapped_column(
        Enum(LeadStatus), nullable=False, default=LeadStatus.new
    )
    score: Mapped[int] = mapped_column(Integer, default=0)  # 0-100
    notes: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow,
        server_default=func.now()
    )

    # Relationships
    tenant: Mapped[Tenant] = relationship("Tenant", back_populates="leads")
    owner: Mapped[User] = relationship("User", back_populates="leads")
    activities: Mapped[list[Activity]] = relationship("Activity", back_populates="lead")
    lead_notes: Mapped[list[Note]] = relationship(
        "Note",
        primaryjoin="and_(Note.entity_type=='lead', foreign(Note.entity_id)==Lead.id)",
        viewonly=True,
    )


# ── Customer ──────────────────────────────────────────────────────────────────

class Customer(Base):
    """A converted lead / existing customer."""
    __tablename__ = "customers"
    __table_args__ = (
        Index("ix_customers_tenant_id", "tenant_id"),
        Index("ix_customers_email", "email"),
        Index("ix_customers_tenant_industry", "tenant_id", "industry"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    email: Mapped[str] = mapped_column(String(320), nullable=True)
    phone: Mapped[str] = mapped_column(String(50), nullable=True)
    company: Mapped[str] = mapped_column(String(200), nullable=True)
    industry: Mapped[str] = mapped_column(String(100), nullable=True)
    website: Mapped[str] = mapped_column(String(500), nullable=True)
    address: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow,
        server_default=func.now()
    )

    # Relationships
    tenant: Mapped[Tenant] = relationship("Tenant", back_populates="customers")
    deals: Mapped[list[Deal]] = relationship("Deal", back_populates="customer")
    activities: Mapped[list[Activity]] = relationship("Activity", back_populates="customer")
    interactions: Mapped[list[Interaction]] = relationship(
        "Interaction", back_populates="customer"
    )


# ── Deal ──────────────────────────────────────────────────────────────────────

class Deal(Base):
    """
    Sales opportunity / deal.
    amount: stored as Numeric(15,2) for financial accuracy.
           NEVER use float for money in production.
    probability: 0–100 set by application, not LLM.
    """
    __tablename__ = "deals"
    __table_args__ = (
        Index("ix_deals_tenant_id", "tenant_id"),
        Index("ix_deals_owner_id", "owner_id"),
        Index("ix_deals_stage", "stage"),
        Index("ix_deals_tenant_stage", "tenant_id", "stage"),
        Index("ix_deals_close_date", "expected_close_date"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    owner_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    customer_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("customers.id", ondelete="SET NULL"), nullable=True
    )
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    amount: Mapped[float] = mapped_column(Numeric(15, 2), nullable=True)
    stage: Mapped[DealStage] = mapped_column(
        Enum(DealStage), nullable=False, default=DealStage.prospecting
    )
    probability: Mapped[int] = mapped_column(Integer, default=0)    # 0-100
    expected_close_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    description: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow,
        server_default=func.now()
    )

    # Relationships
    tenant: Mapped[Tenant] = relationship("Tenant", back_populates="deals")
    owner: Mapped[User] = relationship("User", back_populates="deals")
    customer: Mapped[Customer] = relationship("Customer", back_populates="deals")


# ── Activity ──────────────────────────────────────────────────────────────────

class Activity(Base):
    """Calls, emails, meetings, tasks logged against a lead or customer."""
    __tablename__ = "activities"
    __table_args__ = (
        Index("ix_activities_tenant_id", "tenant_id"),
        Index("ix_activities_user_id", "user_id"),
        Index("ix_activities_lead_id", "lead_id"),
        Index("ix_activities_customer_id", "customer_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    lead_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("leads.id", ondelete="SET NULL"), nullable=True
    )
    customer_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("customers.id", ondelete="SET NULL"), nullable=True
    )
    type: Mapped[ActivityType] = mapped_column(Enum(ActivityType), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    due_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[ActivityStatus] = mapped_column(
        Enum(ActivityStatus), nullable=False, default=ActivityStatus.pending
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )

    # Relationships
    user: Mapped[User] = relationship("User", back_populates="activities")
    lead: Mapped[Lead] = relationship("Lead", back_populates="activities")
    customer: Mapped[Customer] = relationship("Customer", back_populates="activities")


# ── Product ───────────────────────────────────────────────────────────────────

class Product(Base):
    """Products/services offered. Used in knowledge base and deal context."""
    __tablename__ = "products"
    __table_args__ = (
        Index("ix_products_tenant_id", "tenant_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    price: Mapped[float] = mapped_column(Numeric(15, 2), nullable=True)
    category: Mapped[str] = mapped_column(String(100), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


# ── Interaction ───────────────────────────────────────────────────────────────

class Interaction(Base):
    """Customer-facing communication log (email threads, call notes, etc.)."""
    __tablename__ = "interactions"
    __table_args__ = (
        Index("ix_interactions_tenant_customer", "tenant_id", "customer_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    customer_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("customers.id", ondelete="CASCADE"), nullable=False
    )
    channel: Mapped[InteractionChannel] = mapped_column(
        Enum(InteractionChannel), nullable=False
    )
    message: Mapped[str] = mapped_column(Text, nullable=False)
    direction: Mapped[str] = mapped_column(String(20), default="inbound")  # inbound | outbound
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )

    # Relationships
    customer: Mapped[Customer] = relationship("Customer", back_populates="interactions")


# ── Note ──────────────────────────────────────────────────────────────────────

class Note(Base):
    """
    Polymorphic notes attached to any CRM entity.
    entity_type: "lead" | "customer" | "deal"
    entity_id: the UUID of that record.
    WHY polymorphic: avoids separate notes tables for every entity type.
    """
    __tablename__ = "notes"
    __table_args__ = (
        Index("ix_notes_tenant_entity", "tenant_id", "entity_type", "entity_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(36), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )

    # Relationships
    user: Mapped[User] = relationship("User", back_populates="notes")
