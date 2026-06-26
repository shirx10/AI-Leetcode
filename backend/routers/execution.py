"""
backend/routers/execution.py
=============================
POST /api/run-tests

Runs user code against a problem's test cases and returns the raw
evaluation result without invoking the LangGraph agent.

Use this endpoint for the "Run" button (fast feedback loop).
Use /api/review-code for the full agent critique.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.models import (
    EvaluationResult,
    RunTestsRequest,
    RunTestsResponse,
    TestCaseResult,
)
from backend.services.data_vault import get_problem_by_id
from backend.services.executor import evaluate_code

router = APIRouter(prefix="/api", tags=["execution"])


def _to_eval_result(raw: dict) -> EvaluationResult:
    return EvaluationResult(
        passed=raw["passed"],
        passed_count=raw["passed_count"],
        total=raw["total"],
        summary=raw["summary"],
        results=[
            TestCaseResult(
                passed=r["passed"],
                input=r.get("input"),
                expected=r.get("expected"),
                actual=r.get("actual"),
                error=r.get("error"),
            )
            for r in raw.get("results", [])
        ],
    )


@router.post("/run-tests", response_model=RunTestsResponse)
async def run_tests(body: RunTestsRequest):
    """
    Execute user code against all hidden test cases for a problem.

    Returns detailed per-case results immediately — no LLM call.
    Fast path for the Submit button when you want test output without
    waiting for the Critic Agent.
    """
    problem = get_problem_by_id(body.problem_id)
    if not problem:
        raise HTTPException(
            status_code=404,
            detail=f"Problem '{body.problem_id}' not found.",
        )

    raw_eval = evaluate_code(body.code, problem)

    return RunTestsResponse(
        problem_id=body.problem_id,
        evaluation=_to_eval_result(raw_eval),
    )
