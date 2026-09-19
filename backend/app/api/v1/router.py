"""
API v1 Router — wires all endpoint modules into a single prefix.

WHY VERSIONED ROUTER:
  /api/v1/ allows future /api/v2/ without breaking existing integrations.
  Each endpoint module owns its own path + tags.
"""
from fastapi import APIRouter

from app.api.v1.endpoints import auth, leads, deals, customers, chat

api_router = APIRouter()

api_router.include_router(auth.router,      prefix="/auth",      tags=["Authentication"])
api_router.include_router(leads.router,     prefix="/leads",     tags=["Leads"])
api_router.include_router(deals.router,     prefix="/deals",     tags=["Deals"])
api_router.include_router(customers.router, prefix="/customers", tags=["Customers"])
api_router.include_router(chat.router,      prefix="/agent",     tags=["AI Agent"])
