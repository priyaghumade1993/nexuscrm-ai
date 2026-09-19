"""
MongoDB connection for conversation history, agent execution metadata, and audit logs.

WHY MongoDB for this (and not PostgreSQL):
  Conversation history is semi-structured — messages vary in length,
  tool calls vary in shape, agent traces vary in depth.
  MongoDB's document model stores this naturally without schema migrations.

  PostgreSQL is optimal for structured CRM data (leads, deals, customers)
  where relationships, foreign keys, and ACID transactions matter.

  MongoDB is optimal for time-series-like append-heavy writes (chat messages,
  audit events) and for schema-flexible agent execution metadata.

WHY NOT duplicate the CRM data in MongoDB:
  Single source of truth. The CRM data lives in PostgreSQL.
  MongoDB only stores what doesn't fit the relational model.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.config import settings

logger = logging.getLogger(__name__)


class MongoManager:
    """Manages the MongoDB async client lifecycle."""

    client: Optional[AsyncIOMotorClient] = None
    db: Optional[AsyncIOMotorDatabase] = None

    @classmethod
    async def connect(cls) -> None:
        try:
            cls.client = AsyncIOMotorClient(
                settings.mongo_uri,
                serverSelectionTimeoutMS=5000,
            )
            cls.db = cls.client[settings.mongo_db]
            # Verify connection
            await cls.client.admin.command("ping")
            await cls._ensure_indexes()
            logger.info("MongoDB connected: %s/%s", settings.mongo_uri, settings.mongo_db)
        except Exception as e:
            logger.warning("MongoDB not available: %s — agent history disabled", e)
            cls.client = None
            cls.db = None

    @classmethod
    async def disconnect(cls) -> None:
        if cls.client:
            cls.client.close()
            cls.client = None
            cls.db = None

    @classmethod
    async def _ensure_indexes(cls) -> None:
        """Create MongoDB indexes for performance and tenant isolation."""
        if cls.db is None:
            return
        # conversation_history indexes
        await cls.db.conversation_history.create_index(
            [("tenant_id", 1), ("session_id", 1), ("created_at", -1)]
        )
        await cls.db.conversation_history.create_index([("user_id", 1)])

        # agent_executions indexes
        await cls.db.agent_executions.create_index(
            [("tenant_id", 1), ("execution_id", 1)]
        )
        await cls.db.agent_executions.create_index([("created_at", -1)])

        # audit_events indexes
        await cls.db.audit_events.create_index([("tenant_id", 1), ("created_at", -1)])
        await cls.db.audit_events.create_index([("user_id", 1)])

    @classmethod
    def is_available(cls) -> bool:
        return cls.db is not None


async def get_mongo_db() -> Optional[AsyncIOMotorDatabase]:
    """Dependency — returns None gracefully if Mongo is unavailable."""
    return MongoManager.db


# ── Conversation history helpers ──────────────────────────────────────────────

async def save_message(
    *,
    tenant_id: str,
    user_id: str,
    session_id: str,
    role: str,                # "user" | "assistant" | "tool"
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    if not MongoManager.is_available():
        return
    doc = {
        "tenant_id": tenant_id,
        "user_id": user_id,
        "session_id": session_id,
        "role": role,
        "content": content,
        "metadata": metadata or {},
        "created_at": datetime.now(timezone.utc),
    }
    await MongoManager.db.conversation_history.insert_one(doc)


async def get_conversation_history(
    *, tenant_id: str, session_id: str, limit: int = 20
) -> List[Dict[str, Any]]:
    if not MongoManager.is_available():
        return []
    cursor = (
        MongoManager.db.conversation_history
        .find({"tenant_id": tenant_id, "session_id": session_id})
        .sort("created_at", 1)
        .limit(limit)
    )
    return await cursor.to_list(length=limit)


async def save_agent_execution(
    *,
    execution_id: str,
    tenant_id: str,
    user_id: str,
    input_text: str,
    intent: str,
    steps: List[Dict[str, Any]],
    output: str,
    latency_ms: int,
    success: bool,
    error: Optional[str] = None,
) -> None:
    if not MongoManager.is_available():
        return
    doc = {
        "execution_id": execution_id,
        "tenant_id": tenant_id,
        "user_id": user_id,
        "input": input_text,
        "intent": intent,
        "steps": steps,
        "output": output,
        "latency_ms": latency_ms,
        "success": success,
        "error": error,
        "created_at": datetime.now(timezone.utc),
    }
    await MongoManager.db.agent_executions.insert_one(doc)


async def save_audit_event(
    *,
    tenant_id: str,
    user_id: str,
    action: str,
    resource_type: str,
    resource_id: Optional[str],
    details: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None,
) -> None:
    if not MongoManager.is_available():
        return
    doc = {
        "tenant_id": tenant_id,
        "user_id": user_id,
        "action": action,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "details": details or {},
        "ip_address": ip_address,
        "created_at": datetime.now(timezone.utc),
    }
    await MongoManager.db.audit_events.insert_one(doc)
