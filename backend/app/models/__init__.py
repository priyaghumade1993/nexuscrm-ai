"""
SQLAlchemy ORM models for NexusCRM AI.

WHY SQLAlchemy:
  - Industry standard ORM for Python
  - Native async support (SQLAlchemy 2.0)
  - Alembic integration for safe schema migrations
  - Type-safe queries with mapped columns (2.0 style)
  - Relationship management and lazy/eager loading control

WHY PostgreSQL:
  - ACID transactions for financial data (deals, amounts)
  - Foreign key constraints for data integrity
  - Advanced indexes (partial, composite, GIN for JSON)
  - Row-level security capability (tenant isolation)
  - Full-text search if needed later
  - JSON/JSONB for flexible metadata
"""
from .crm import (
    Tenant,
    User,
    Lead,
    Customer,
    Deal,
    Activity,
    Product,
    Interaction,
    Note,
)

__all__ = [
    "Tenant",
    "User",
    "Lead",
    "Customer",
    "Deal",
    "Activity",
    "Product",
    "Interaction",
    "Note",
]
