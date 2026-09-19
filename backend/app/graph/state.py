"""
LangGraph State — the shared data structure passed between all graph nodes.

WHY LANGGRAPH vs a simple LangChain chain:
  LangChain chains are linear (A → B → C).
  LangGraph supports CONDITIONAL ROUTING — based on intent, route to
  different agents. It also supports cycles (retry on tool failure),
  checkpointing (resume a broken run), and human-in-the-loop pauses
  (e.g., waiting for delete confirmation).

WHAT IS STATE:
  In LangGraph, each node receives the current State and returns a
  partial update. The framework merges updates automatically.
  State is the "working memory" of the agent for one request.
"""
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional
from pydantic import BaseModel

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(BaseModel):
    """
    The complete state of one agent execution.
    Annotated[list, add_messages] tells LangGraph to APPEND new messages
    rather than replace the entire list.
    """
    # Input
    user_message: str = ""
    tenant_id: str = ""
    user_id: str = ""
    user_role: str = ""
    session_id: str = ""
    confirm_destructive: bool = False

    # Derived during workflow
    intent: Optional[str] = None       # classified intent
    routed_agent: Optional[str] = None # which specialist handles it
    messages: Annotated[List[BaseMessage], add_messages] = []

    # Tool execution results
    tool_calls: List[Dict[str, Any]] = []
    tool_results: List[str] = []

    # RAG
    rag_context: Optional[str] = None
    rag_sources: List[str] = []

    # Auth
    is_authorized: bool = True
    auth_failure_reason: Optional[str] = None

    # Output
    final_response: Optional[str] = None
    requires_confirmation: bool = False
    confirmation_message: Optional[str] = None

    # Guardrails
    injection_detected: bool = False
    guardrail_failure: Optional[str] = None

    # Observability
    execution_id: str = ""
    step_count: int = 0
    errors: List[str] = []

    class Config:
        arbitrary_types_allowed = True
