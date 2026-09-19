"""
NexusCRM AI — FastAPI Application Entry Point.

STARTUP SEQUENCE:
  1. Load settings (validates env vars via Pydantic)
  2. Connect to MongoDB (graceful degradation if unavailable)
  3. Compile LangGraph workflow (singleton cached graph)
  4. Register all API routers under /api/v1

WHY LIFESPAN OVER @app.on_event:
  @app.on_event("startup") is deprecated in FastAPI 0.93+.
  Lifespan context manager is the idiomatic replacement and
  supports both startup AND shutdown in one place.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import get_settings
from app.api.v1.router import api_router

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Logging Configuration ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)


# ── Lifespan: startup + shutdown ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup:
      - Connect MongoDB (non-blocking; app works without it)
      - Compile LangGraph workflow (pre-warm; avoids cold start on first request)

    Shutdown:
      - Disconnect MongoDB cleanly
    """
    # ── STARTUP ──────────────────────────────────────────────────────────────
    logger.info("NexusCRM AI starting up…")

    # 1. MongoDB
    try:
        from app.database.mongodb import mongo_manager
        await mongo_manager.connect()
        logger.info("MongoDB connected")
    except Exception as exc:
        logger.warning("MongoDB unavailable (conversation history disabled): %s", exc)

    # 2. Pre-compile the LangGraph workflow so first request is fast
    try:
        from app.graph.workflow import get_workflow
        get_workflow()
        logger.info("LangGraph workflow compiled and cached")
    except Exception as exc:
        logger.warning("LangGraph pre-compile failed (will retry on first request): %s", exc)

    logger.info(
        "NexusCRM AI ready | env=%s | openai=%s | langsmith=%s | vector_store=%s",
        settings.environment,
        "configured" if settings.openai_configured else "NOT configured",
        "configured" if settings.langsmith_configured else "not configured",
        settings.vector_store,
    )

    yield  # ── Application runs ──────────────────────────────────────────────

    # ── SHUTDOWN ─────────────────────────────────────────────────────────────
    logger.info("NexusCRM AI shutting down…")
    try:
        from app.database.mongodb import mongo_manager
        await mongo_manager.disconnect()
        logger.info("MongoDB disconnected")
    except Exception as exc:
        logger.warning("MongoDB disconnect error: %s", exc)


# ── FastAPI Application ───────────────────────────────────────────────────────
app = FastAPI(
    title="NexusCRM AI",
    description=(
        "Production-grade Agentic AI CRM with LangGraph multi-agent workflow, "
        "RAG knowledge base, multi-tenant RBAC, and full observability."
    ),
    version="1.0.0",
    docs_url="/docs" if not settings.is_production else None,   # hide Swagger in prod
    redoc_url="/redoc" if not settings.is_production else None,
    lifespan=lifespan,
)

# ── CORS Middleware ───────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Router ────────────────────────────────────────────────────────────────
app.include_router(api_router, prefix="/api/v1")


# ── Health Check (no auth required) ──────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Lightweight health probe for load balancers and Kubernetes liveness checks.
    Returns service status and configuration flags (never secrets).
    """
    from app.database.base import engine
    db_ok = False
    try:
        async with engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy", fromlist=["text"]).text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    return {
        "status": "healthy" if db_ok else "degraded",
        "service": "nexuscrm-ai",
        "version": "1.0.0",
        "database": "connected" if db_ok else "unavailable",
        "openai": "configured" if settings.openai_configured else "not configured",
        "langsmith": "configured" if settings.langsmith_configured else "not configured",
        "vector_store": settings.vector_store,
    }


# ── Root Redirect ─────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "NexusCRM AI API",
        "docs": "/docs",
        "health": "/health",
        "api": "/api/v1",
    }
