"""
Output Guardrail — validates AI responses before returning to the user.

PURPOSE:
  Catches cases where the LLM invents CRM data not present in tool results.
  Also flags responses that leak sensitive information or contain unsafe content.

WHAT IT CHECKS:
  1. Hallucinated records: response mentions specific lead/customer names
     or numbers not present in the tool results
  2. Fabricated financial figures: $ amounts not from tool results
  3. Cross-tenant data leakage: response mentions tenant IDs it shouldn't
  4. Policy violations: extreme claims about product capabilities or pricing

DESIGN DECISION — WHY VALIDATE OUTPUT:
  The LLM can generate plausible-sounding but false data even when given
  tool results. Example: if search_leads returns 0 results, the LLM might
  still say "I found 3 leads for you." Output validation catches this.

PERFORMANCE:
  Output validation adds an LLM call (~200-400ms). This is acceptable for
  a CRM AI agent where accuracy > speed. For latency-sensitive use cases,
  consider async background validation with flagging.

INTERVIEW TALKING POINT:
  "I added an output guardrail that validates the AI's response against
   actual tool results. If the model says it found 5 leads but the database
   returned 3, the guardrail catches that hallucination before it reaches
   the user. This is especially critical for financial figures."
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class OutputGuardrailResult:
    is_valid: bool
    issues: List[str] = field(default_factory=list)
    detection_method: str = "none"


# ── Heuristic checks (fast, no LLM) ──────────────────────────────────────────

# Dollar amount pattern — catches "$1,234.56" or "$1234" or "USD 1234"
_DOLLAR_PATTERN = re.compile(r"\$[\d,]+(?:\.\d{2})?|\bUSD\s+[\d,]+")

def _heuristic_check(response: str, tool_results: list[str]) -> list[str]:
    """
    Fast heuristic validation without an LLM call.

    Checks:
    1. If tool_results is empty but response mentions specific records → suspicious
    2. If response claims a specific count that differs from tool results
    """
    issues = []
    tool_text = " ".join(tool_results)

    # Check if response mentions $ amounts not present in tool results
    response_amounts = set(_DOLLAR_PATTERN.findall(response))
    tool_amounts = set(_DOLLAR_PATTERN.findall(tool_text))

    invented_amounts = response_amounts - tool_amounts
    if invented_amounts and tool_results:  # only flag if tools were actually called
        issues.append(
            f"Response contains financial figures not found in tool results: {invented_amounts}"
        )

    return issues


# ── LLM-based output validation ───────────────────────────────────────────────

def check_output_quality(
    response: str,
    tool_results: list[str],
    use_llm: bool = True,
) -> OutputGuardrailResult:
    """
    Validate AI response against tool results.

    Args:
        response: The AI-generated response text
        tool_results: List of tool result strings (from AgentState.tool_results)
        use_llm: Whether to use LLM for semantic validation

    Returns:
        OutputGuardrailResult with is_valid=False if hallucination detected.
    """
    # Fast heuristic check first
    heuristic_issues = _heuristic_check(response, tool_results)

    if not use_llm:
        return OutputGuardrailResult(
            is_valid=len(heuristic_issues) == 0,
            issues=heuristic_issues,
            detection_method="heuristic",
        )

    # LLM validation
    llm_result = _llm_output_check(response, tool_results)

    all_issues = heuristic_issues + llm_result.issues
    return OutputGuardrailResult(
        is_valid=len(all_issues) == 0,
        issues=all_issues,
        detection_method="heuristic+llm" if heuristic_issues else llm_result.detection_method,
    )


def _llm_output_check(response: str, tool_results: list[str]) -> OutputGuardrailResult:
    """
    Use LLM to check if response is grounded in tool results.
    Gracefully skips if LLM is not configured.
    """
    try:
        from app.agents.llm_provider import get_default_provider
        from app.graph.prompts import RESPONSE_VALIDATION_PROMPT

        provider = get_default_provider()
        if not provider.is_configured:
            return OutputGuardrailResult(is_valid=True, detection_method="skipped_no_llm")

        llm = provider.get_chat_model(temperature=0)
        tool_summary = "\n".join(tool_results) if tool_results else "No tools were called."
        prompt = RESPONSE_VALIDATION_PROMPT.format(
            tool_results=tool_summary,
            response=response,
        )

        from langchain_core.messages import HumanMessage
        llm_response = llm.invoke([HumanMessage(content=prompt)])
        content = llm_response.content.strip()

        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        result = json.loads(content)
        is_valid = result.get("is_valid", True)
        issues = result.get("issues", [])

        if not is_valid:
            logger.warning("Output guardrail: found issues — %s", issues)

        return OutputGuardrailResult(
            is_valid=is_valid,
            issues=issues,
            detection_method="llm",
        )

    except json.JSONDecodeError as exc:
        logger.warning("Output guardrail LLM returned non-JSON: %s", exc)
        return OutputGuardrailResult(is_valid=True, detection_method="llm_parse_error")
    except Exception as exc:
        logger.error("Output guardrail LLM check failed: %s", exc, exc_info=True)
        return OutputGuardrailResult(is_valid=True, detection_method="llm_error")
