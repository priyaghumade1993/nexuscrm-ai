from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None        # for conversation continuity
    confirm_destructive: bool = False        # explicit user confirmation for DELETE ops


class ToolCallLog(BaseModel):
    tool: str
    arguments: Dict[str, Any]
    result: str
    success: bool


class ChatResponse(BaseModel):
    response: str
    intent: Optional[str] = None
    agent: Optional[str] = None             # which specialist handled it
    tools_used: List[ToolCallLog] = []
    session_id: Optional[str] = None
    execution_id: Optional[str] = None
    latency_ms: Optional[int] = None
    requires_confirmation: bool = False     # True = needs confirm_destructive=True
    confirmation_message: Optional[str] = None
