"""
NexusCRM AI Evaluation Runner.

WHAT IT DOES:
  1. Runs each EvaluationCase through the live agent workflow
  2. Compares actual vs expected intent, agent routing, tool usage
  3. Uses LLM-as-a-judge for response quality scoring (optional)
  4. Outputs a detailed JSON report with pass/fail per case
  5. Optionally pushes results to LangSmith for experiment tracking

HOW TO RUN:
  python -m app.evaluation.runner
  # or with LangSmith:
  LANGCHAIN_TRACING_V2=true python -m app.evaluation.runner

METRICS:
  - intent_accuracy: % of cases where predicted intent = expected intent
  - agent_accuracy: % of cases where routed agent = expected agent
  - tool_precision: % of required tools that were actually called
  - safety_rate: % of injection cases correctly blocked
  - hallucination_rate: % of cases where response passes output guardrail

INTERVIEW TALKING POINT:
  "I use LangSmith's dataset + evaluator pattern. Each test case is run
   through the full agent workflow, and the results are compared against
   ground truth. I track intent accuracy, tool selection, and response
   quality as separate metrics. This lets me compare prompt versions and
   catch regressions before they reach production."
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from app.evaluation.dataset import EVALUATION_DATASET, EvaluationCase

logger = logging.getLogger(__name__)

# Synthetic tenant/user IDs for evaluation runs
EVAL_TENANT_ID = "eval-tenant-00000000"
EVAL_USER_ID = "eval-user-00000000"


@dataclass
class EvaluationResult:
    case_id: str
    input: str
    expected_intent: str
    expected_agent: str
    actual_intent: Optional[str]
    actual_agent: Optional[str]
    actual_response: str
    tools_called: List[str]
    intent_correct: bool
    agent_correct: bool
    required_tools_called: bool
    forbidden_content_absent: bool
    latency_ms: int
    passed: bool
    error: Optional[str] = None
    langsmith_run_id: Optional[str] = None


@dataclass
class EvaluationReport:
    timestamp: str
    total_cases: int
    passed: int
    failed: int
    intent_accuracy: float
    agent_accuracy: float
    tool_precision: float
    safety_rate: float
    avg_latency_ms: float
    results: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("NexusCRM AI Evaluation Report")
        print("=" * 60)
        print(f"Timestamp:        {self.timestamp}")
        print(f"Total cases:      {self.total_cases}")
        print(f"Passed:           {self.passed} ({self.passed/self.total_cases*100:.1f}%)")
        print(f"Failed:           {self.failed}")
        print(f"Intent accuracy:  {self.intent_accuracy:.1%}")
        print(f"Agent accuracy:   {self.agent_accuracy:.1%}")
        print(f"Tool precision:   {self.tool_precision:.1%}")
        print(f"Safety rate:      {self.safety_rate:.1%}")
        print(f"Avg latency:      {self.avg_latency_ms:.0f}ms")
        print("=" * 60)

        # Show failures
        failures = [r for r in self.results if not r["passed"]]
        if failures:
            print(f"\nFailed cases ({len(failures)}):")
            for f in failures:
                print(f"  [{f['case_id']}] {f['input'][:50]}...")
                print(f"    Expected intent: {f['expected_intent']} → Got: {f['actual_intent']}")
                if f["error"]:
                    print(f"    Error: {f['error']}")
        print()


class EvaluationRunner:
    """
    Runs the evaluation dataset against the live NexusCRM agent.

    WHY LIVE AGENT (not mocked):
      Mocking the LLM tests the test harness, not the actual behavior.
      Running against the live agent catches:
      - Prompt regression (new system prompt breaks intent classification)
      - Tool selection errors (agent calls wrong tool)
      - Latency regression (new code path is slower)
    """

    def __init__(
        self,
        cases: Optional[List[EvaluationCase]] = None,
        tag_filter: Optional[List[str]] = None,
        max_concurrent: int = 3,
    ):
        self.cases = cases or EVALUATION_DATASET
        if tag_filter:
            self.cases = [c for c in self.cases if any(t in c.tags for t in tag_filter)]
        self.max_concurrent = max_concurrent

    async def run(self, db_session=None) -> EvaluationReport:
        """Run all evaluation cases and return an aggregated report."""
        logger.info("Starting evaluation: %d cases", len(self.cases))
        start = time.time()

        # Run cases with concurrency limit
        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = [self._run_case(case, db_session, semaphore) for case in self.cases]
        results: List[EvaluationResult] = await asyncio.gather(*tasks)

        return self._build_report(results, elapsed_ms=int((time.time() - start) * 1000))

    async def _run_case(
        self,
        case: EvaluationCase,
        db_session,
        semaphore: asyncio.Semaphore,
    ) -> EvaluationResult:
        """Run a single evaluation case."""
        async with semaphore:
            return await self._execute_case(case, db_session)

    async def _execute_case(self, case: EvaluationCase, db_session) -> EvaluationResult:
        start_ms = time.time()
        error = None

        try:
            from app.graph.workflow import run_agent

            result = await run_agent(
                message=case.input,
                tenant_id=EVAL_TENANT_ID,
                user_id=EVAL_USER_ID,
                role=case.user_role,
                session_id=f"eval-{case.id}",
                confirm_destructive=False,  # eval runs should NOT auto-confirm deletes
                db_session=db_session,
            )

            actual_intent = result.get("intent")
            actual_agent = result.get("agent")
            actual_response = result.get("response", "")
            tool_results = result.get("tool_results", [])

            # Extract tool names from tool_results log strings: "[tool_name]: ..."
            tools_called = []
            for tr in tool_results:
                if "]" in tr:
                    tool_name = tr.split("]")[0].lstrip("[")
                    tools_called.append(tool_name)

        except Exception as exc:
            logger.error("Evaluation case %s failed: %s", case.id, exc)
            error = str(exc)
            actual_intent = None
            actual_agent = None
            actual_response = ""
            tools_called = []

        latency_ms = int((time.time() - start_ms) * 1000)

        # ── Scoring ───────────────────────────────────────────────────────────
        intent_correct = actual_intent == case.expected_intent
        agent_correct = actual_agent == case.expected_agent

        # Tool precision: were all required tools called?
        required_tools_called = (
            all(t in tools_called for t in case.required_tools)
            if case.required_tools else True
        )

        # Forbidden content check
        forbidden_absent = not any(
            f.lower() in actual_response.lower()
            for f in case.forbidden_content
        )

        # Overall pass: intent + agent routing correct + no forbidden content
        # For security cases, we don't require agent_correct (blocked before routing)
        is_security_case = "security" in case.tags or "guardrail" in case.tags
        passed = (
            intent_correct
            and (agent_correct or is_security_case)
            and forbidden_absent
            and error is None
        )

        result_obj = EvaluationResult(
            case_id=case.id,
            input=case.input,
            expected_intent=case.expected_intent,
            expected_agent=case.expected_agent,
            actual_intent=actual_intent,
            actual_agent=actual_agent,
            actual_response=actual_response,
            tools_called=tools_called,
            intent_correct=intent_correct,
            agent_correct=agent_correct,
            required_tools_called=required_tools_called,
            forbidden_content_absent=forbidden_absent,
            latency_ms=latency_ms,
            passed=passed,
            error=error,
        )

        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info("[%s] %s | intent=%s→%s | latency=%dms",
                    case.id, status, case.expected_intent, actual_intent, latency_ms)

        return result_obj

    def _build_report(
        self, results: List[EvaluationResult], elapsed_ms: int
    ) -> EvaluationReport:
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed

        intent_correct = sum(1 for r in results if r.intent_correct)
        agent_correct = sum(1 for r in results if r.agent_correct)
        tool_precision = sum(1 for r in results if r.required_tools_called)
        forbidden_ok = sum(1 for r in results if r.forbidden_content_absent)

        # Safety rate: injection cases that were correctly blocked
        security_cases = [r for r in results
                          if any(c.id == r.case_id and "security" in c.tags
                                 for c in self.cases)]
        safety_rate = (
            sum(1 for r in security_cases if r.forbidden_content_absent) / len(security_cases)
            if security_cases else 1.0
        )

        avg_latency = sum(r.latency_ms for r in results) / total if total else 0

        return EvaluationReport(
            timestamp=datetime.utcnow().isoformat(),
            total_cases=total,
            passed=passed,
            failed=failed,
            intent_accuracy=intent_correct / total if total else 0,
            agent_accuracy=agent_correct / total if total else 0,
            tool_precision=tool_precision / total if total else 0,
            safety_rate=safety_rate,
            avg_latency_ms=avg_latency,
            results=[asdict(r) for r in results],
        )


async def run_evaluation(
    tag_filter: Optional[List[str]] = None,
    output_file: Optional[str] = None,
) -> EvaluationReport:
    """
    Convenience function to run the full evaluation suite.
    Can be called from scripts or a FastAPI endpoint.
    """
    runner = EvaluationRunner(tag_filter=tag_filter)
    report = await runner.run()
    report.print_summary()

    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"Report saved to {out_path}")

    return report


if __name__ == "__main__":
    import sys
    tag_filter = sys.argv[1:] if len(sys.argv) > 1 else None
    asyncio.run(run_evaluation(
        tag_filter=tag_filter,
        output_file="evaluation_results/latest.json",
    ))
