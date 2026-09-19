"""Chat endpoint — connects FastAPI to the LangGraph agent workflow."""
import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import get_current_user
from app.database import get_db
from app.database.mongodb import save_message, save_agent_execution
from app.graph.workflow import run_agent
from app.models.crm import User
from app.schemas.chat import ChatRequest, ChatResponse, ToolCallLog

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Main AI chat endpoint.

    Flow:
    1. Save user message to MongoDB (conversation history)
    2. Run LangGraph workflow
    3. Save agent response to MongoDB
    4. Save execution metadata
    5. Return structured response

    WHY separate from the CRUD endpoints:
      CRUD endpoints are deterministic REST operations.
      This endpoint invokes an intelligent agent that reasons about what
      the user wants and calls appropriate tools — not all requests map
      to a single API call.
    """
    # Save user message
    await save_message(
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
        session_id=request.session_id or "new",
        role="user",
        content=request.message,
    )

    # Run the agentic workflow
    result = await run_agent(
        message=request.message,
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
        role=current_user.role.value,
        session_id=request.session_id,
        confirm_destructive=request.confirm_destructive,
        db_session=db,
    )

    # Save assistant response
    await save_message(
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
        session_id=result["session_id"],
        role="assistant",
        content=result["response"],
        metadata={
            "intent": result.get("intent"),
            "agent": result.get("agent"),
            "execution_id": result.get("execution_id"),
            "latency_ms": result.get("latency_ms"),
        },
    )

    # Save execution metadata for observability
    await save_agent_execution(
        execution_id=result["execution_id"],
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
        input_text=request.message,
        intent=result.get("intent") or "unknown",
        steps=[{"tool_result": tr} for tr in result.get("tool_results", [])],
        output=result["response"],
        latency_ms=result.get("latency_ms", 0),
        success=len(result.get("errors", [])) == 0,
        error=result["errors"][0] if result.get("errors") else None,
    )

    # Build tool call logs
    tool_logs = [
        ToolCallLog(
            tool=tr.split("]")[0].lstrip("[") if "]" in tr else "unknown",
            arguments={},
            result=tr.split("]: ", 1)[1] if "]: " in tr else tr,
            success=not tr.startswith("Tool error"),
        )
        for tr in result.get("tool_results", [])
    ]

    return ChatResponse(
        response=result["response"],
        intent=result.get("intent"),
        agent=result.get("agent"),
        tools_used=tool_logs,
        session_id=result["session_id"],
        execution_id=result["execution_id"],
        latency_ms=result.get("latency_ms"),
        requires_confirmation=result.get("requires_confirmation", False),
        confirmation_message=result.get("confirmation_message"),
    )
