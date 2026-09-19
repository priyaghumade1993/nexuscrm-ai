"""
NexusCRM AI — LangGraph Workflow Builder.

This assembles all nodes and edges into the compiled graph.

WHY LANGGRAPH:
  LangGraph provides:
  1. Stateful graph execution — state is shared and updated across nodes.
  2. Conditional routing — different paths based on intent/auth/guardrails.
  3. Checkpointing — graphs can be resumed after failure.
  4. Built-in LangSmith tracing — every node execution is traced automatically.
  5. Human-in-the-loop — can pause at any node waiting for user input.

WHY NOT A SIMPLE LANGCHAIN CHAIN:
  Chains are linear. Our workflow branches:
  - Guardrail failure → skip everything → error response
  - Auth failure → skip → unauthorized response
  - intent=knowledge → KnowledgeAgent (not SalesAgent)
  - intent=analytics → AnalyticsAgent
  None of this is possible with a simple chain.

GRAPH TOPOLOGY:
  START → guardrail_check ──(injection?)──→ END
                          ↓
                    intent_classifier
                          ↓
                  authorization_validator ──(denied?)──→ END
                          ↓
                       supervisor
                          ↓
          ┌──────────────┼──────────────────┬──────────────┐
          ↓              ↓                  ↓              ↓
     sales_agent  customer_agent  analytics_agent  knowledge_agent
          ↓              ↓                  ↓              ↓
          └──────────────┴──────────────────┴──────────────┘
                          ↓
                         END
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Optional

from langgraph.graph import StateGraph, END

from app.graph.state import AgentState
from app.graph.nodes import (
    guardrail_check,
    intent_classifier,
    authorization_validator,
    supervisor,
    sales_agent,
    customer_agent,
    analytics_agent,
    knowledge_agent,
    email_agent,
    route_after_guardrail,
    route_after_auth,
    route_after_supervisor,
)

logger = logging.getLogger(__name__)


def build_workflow():
    """
    Compile the LangGraph workflow.
    Called once at startup and cached.
    """
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("guardrail_check", guardrail_check)
    builder.add_node("intent_classifier", intent_classifier)
    builder.add_node("authorization_validator", authorization_validator)
    builder.add_node("supervisor", supervisor)
    builder.add_node("sales_agent", sales_agent)
    builder.add_node("customer_agent", customer_agent)
    builder.add_node("analytics_agent", analytics_agent)
    builder.add_node("knowledge_agent", knowledge_agent)
    builder.add_node("email_agent", email_agent)

    # Entry point
    builder.set_entry_point("guardrail_check")

    # Conditional edge: after guardrail check
    builder.add_conditional_edges(
        "guardrail_check",
        route_after_guardrail,
        {
            "end": END,
            "classify": "intent_classifier",
        }
    )

    # Linear: intent → auth
    builder.add_edge("intent_classifier", "authorization_validator")

    # Conditional edge: after auth
    builder.add_conditional_edges(
        "authorization_validator",
        route_after_auth,
        {
            "end": END,
            "supervise": "supervisor",
        }
    )

    # Conditional edge: supervisor → specialist agent
    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "sales_agent": "sales_agent",
            "customer_agent": "customer_agent",
            "analytics_agent": "analytics_agent",
            "knowledge_agent": "knowledge_agent",
            "email_agent": "email_agent",
        }
    )

    # All agents go to END
    for agent_node in ["sales_agent", "customer_agent", "analytics_agent",
                       "knowledge_agent", "email_agent"]:
        builder.add_edge(agent_node, END)

    return builder.compile()


# ── Singleton compiled graph ──────────────────────────────────────────────────
_compiled_graph = None


def get_workflow():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_workflow()
        logger.info("LangGraph workflow compiled successfully")
    return _compiled_graph


# ── Main entry point: run the agent ───────────────────────────────────────────

async def run_agent(
    *,
    message: str,
    tenant_id: str,
    user_id: str,
    role: str,
    session_id: Optional[str] = None,
    confirm_destructive: bool = False,
    db_session: Any = None,
) -> dict:
    """
    Execute the full LangGraph workflow for a user message.
    Returns a structured result dict.
    """
    execution_id = str(uuid.uuid4())
    session_id = session_id or str(uuid.uuid4())
    start_time = time.time()

    initial_state = AgentState(
        user_message=message,
        tenant_id=tenant_id,
        user_id=user_id,
        user_role=role,
        session_id=session_id,
        confirm_destructive=confirm_destructive,
        execution_id=execution_id,
    )
    # Attach db_session as a non-model attribute (not in Pydantic schema)
    initial_state._db_session = db_session

    try:
        graph = get_workflow()
        final_state = await graph.ainvoke(initial_state)

        latency_ms = int((time.time() - start_time) * 1000)
        logger.info(
            "Agent execution complete: id=%s intent=%s agent=%s latency=%dms",
            execution_id, final_state.get("intent"), final_state.get("routed_agent"), latency_ms
        )

        return {
            "response": final_state.get("final_response") or "I could not process your request.",
            "intent": final_state.get("intent"),
            "agent": final_state.get("routed_agent"),
            "tool_results": final_state.get("tool_results", []),
            "session_id": session_id,
            "execution_id": execution_id,
            "latency_ms": latency_ms,
            "requires_confirmation": final_state.get("requires_confirmation", False),
            "confirmation_message": final_state.get("confirmation_message"),
            "errors": final_state.get("errors", []),
        }

    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error("Agent workflow error: %s", e, exc_info=True)
        return {
            "response": f"I encountered an unexpected error. Please try again. (ID: {execution_id})",
            "intent": None,
            "agent": None,
            "tool_results": [],
            "session_id": session_id,
            "execution_id": execution_id,
            "latency_ms": latency_ms,
            "requires_confirmation": False,
            "confirmation_message": None,
            "errors": [str(e)],
        }
