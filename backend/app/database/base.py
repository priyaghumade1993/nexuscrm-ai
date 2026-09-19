"""
PostgreSQL async database setup using SQLAlchemy 2.0 + asyncpg.

WHY async:
  FastAPI is async-first. Using asyncpg avoids blocking the event loop
  during DB queries, which is critical for concurrent chat requests
  hitting both the LLM (slow) and the database simultaneously.

WHY SQLAlchemy 2.0:
  Native async support, type-safe ORM, Alembic migrations, and the
  largest ecosystem for production Python database work.
"""
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

# ── Engine ────────────────────────────────────────────────────────────────────
# pool_size / max_overflow: tune for your server's Postgres max_connections.
# echo=False in production — SQL logging is expensive at scale.
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,          # verify connection health before use
    pool_recycle=1800,           # recycle connections every 30 min
)

# ── Session factory ───────────────────────────────────────────────────────────
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,      # keep attributes accessible after commit
    autocommit=False,
    autoflush=False,
)


# ── Declarative base ──────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    """
    All ORM models inherit from this.
    Alembic reads this metadata to auto-generate migrations.
    """
    pass


# ── Dependency injection ──────────────────────────────────────────────────────
async def get_db() -> AsyncSession:
    """
    FastAPI dependency that yields a database session per request.

    WHY yield instead of return:
      The finally block guarantees the session is always closed,
      even if the request handler raises an exception.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
