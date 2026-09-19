"""
Input Guardrail — detects prompt injection and malicious inputs.

TWO-LAYER DEFENSE:
  Layer 1: Fast pattern matching (microseconds, no LLM call needed)
    - Regex patterns for common injection phrases
    - Catches 90%+ of attacks without burning LLM tokens

  Layer 2: LLM-based semantic check (only when patterns don't catch it)
    - The LLM reads the message with a classifier prompt
    - Catches subtle social engineering ("pretend you are a different AI")
    - Returns JSON: {is_safe: bool, reason: str}

WHY BOTH:
  Pattern matching is O(n) and deterministic — fast, cheap, no false negatives
  on known attacks. LLM check catches novel phrasing that patterns miss.
  Together they provide defense in depth.

INTERVIEW TALKING POINT:
  "I implemented a two-layer guardrail: fast regex first for known attack
   signatures, then an LLM semantic check for novel injections. The regex
   runs in microseconds and handles 90%+ of cases; the LLM check only fires
   when the regex doesn't match, keeping latency and cost low."
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ── Layer 1: Fast Pattern Matching ────────────────────────────────────────────
# These are known prompt injection signatures. Patterns are case-insensitive.
INJECTION_PATTERNS = [
    # Override instructions
    r"ignore (your |all )?(previous |prior |above )?instructions",
    r"disregard (your |all )?(previous |prior |above )?instructions",
    r"forget (your |all )?(previous |prior |above )?instructions",
    r"override (your |all )?(previous |prior |above )?instructions",

    # DAN / jailbreak personas
    r"\bdan\b.*mode",
    r"do anything now",
    r"you (are|were|will be) now",
    r"act as (if you are|a )?(different|new|another|unrestricted)",
    r"pretend (you have no|you are|you don.t have) (restrictions|rules|guidelines|limits)",
    r"you have no (restrictions|guidelines|rules|safety|limits)",
    r"jailbreak",

    # Cross-tenant data access
    r"show (me |us )?(all|every|other) (tenant|customer|user|company).s? (data|records|leads|deals)",
    r"access (all|every|other) tenant",
    r"data from (all|every|other) tenant",

    # SQL injection via chat
    r"(execute|run|perform) (a |the )?(sql|query|select|insert|update|delete|drop)",
    r"select \* from",
    r"drop (table|database|index)",
    r"union (all )?select",
    r"; ?drop ",
    r"--\s*(sql|drop|delete|select)",

    # System prompt extraction
    r"(reveal|show|print|display|repeat|output) (your |the )?(system )?prompt",
    r"what (are|were) your instructions",
    r"ignore (the |your )?(system |initial )?prompt",
    r"(above|previous) (prompt|message|instruction)",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


@dataclass
class InputGuardrailResult:
    is_safe: bool
    reason: Optional[str] = None
    detection_method: str = "none"


def check_input_safety(message: str, use_llm: bool = True) -> InputGuardrailResult:
    """
    Run both guardrail layers on the user's input.

    Args:
        message: Raw user message text
        use_llm: Whether to fall back to LLM check if patterns don't match.
                 Set False for unit tests or when OpenAI is not configured.

    Returns:
        InputGuardrailResult with is_safe=False if attack detected.
    """
    # ── Layer 1: Pattern matching ─────────────────────────────────────────────
    for pattern in _COMPILED_PATTERNS:
        match = pattern.search(message)
        if match:
            reason = f"Detected suspicious pattern: '{match.group()}'"
            logger.warning("Guardrail (pattern): blocked message — %s", reason)
            return InputGuardrailResult(
                is_safe=False,
                reason=reason,
                detection_method="pattern",
            )

    # ── Layer 2: LLM semantic check ───────────────────────────────────────────
    if not use_llm:
        return InputGuardrailResult(is_safe=True, detection_method="pattern_only")

    return _llm_safety_check(message)


def _llm_safety_check(message: str) -> InputGuardrailResult:
    """
    Call the LLM with the GUARDRAIL_CHECK_PROMPT to detect semantic attacks.
    Gracefully returns safe=True if LLM is not configured (fail-open for availability).

    WHY FAIL-OPEN:
      If we fail-closed (block when LLM unavailable), every legitimate user query
      fails during LLM downtime. The pattern matching layer already catches the
      most dangerous known attacks. Fail-open is the right tradeoff for a CRM.
      For financial/medical systems, fail-closed would be appropriate.
    """
    try:
        from app.agents.llm_provider import get_default_provider
        from app.graph.prompts import GUARDRAIL_CHECK_PROMPT

        provider = get_default_provider()
        if not provider.is_configured:
            logger.debug("LLM guardrail skipped — OpenAI not configured")
            return InputGuardrailResult(is_safe=True, detection_method="pattern_only")

        llm = provider.get_chat_model(temperature=0)
        prompt = GUARDRAIL_CHECK_PROMPT.format(message=message)

        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()

        # Parse JSON response
        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        result = json.loads(content)
        is_safe = result.get("is_safe", True)
        reason = result.get("reason")

        if not is_safe:
            logger.warning("Guardrail (LLM): blocked message — %s", reason)
            return InputGuardrailResult(
                is_safe=False,
                reason=reason or "LLM detected unsafe content",
                detection_method="llm",
            )

        return InputGuardrailResult(is_safe=True, detection_method="llm")

    except json.JSONDecodeError as exc:
        logger.warning("Guardrail LLM returned non-JSON: %s", exc)
        return InputGuardrailResult(is_safe=True, detection_method="llm_parse_error")
    except Exception as exc:
        logger.error("Guardrail LLM check failed: %s", exc, exc_info=True)
        return InputGuardrailResult(is_safe=True, detection_method="llm_error")
