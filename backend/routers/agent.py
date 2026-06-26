"""
backend/routers/agent.py
=========================
POST /api/review-code

Runs the full LangGraph evaluation + critic loop:
  evaluate_code_node → mastery_node | critic_node

Returns the evaluation result, critic feedback, and the updated
session state (history, mastery_score, session_log) so the client
can persist it without any server-side session storage.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.models import (
    EvaluationResult,
    ReviewCodeRequest,
    ReviewCodeResponse,
    TestCaseResult,
)
from backend.services.data_vault import get_problem_by_id
from backend.graph.builder import eval_graph

router = APIRouter(prefix="/api", tags=["agent"])


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


@router.post("/review-code", response_model=ReviewCodeResponse)
async def review_code(body: ReviewCodeRequest):
    """
    Full agent loop: evaluate code → update mastery → (if failed) run Critic Agent.

    The Gemini API key is read from the X-Gemini-Key request header so it is
    never stored server-side.  Pass it from the Streamlit sidebar input.

    The response includes updated_history, updated_mastery_score, and
    updated_session_log so the stateless client can store them locally
    (e.g. in st.session_state or localStorage) and send them back on the
    next request.
    """
    problem = get_problem_by_id(body.problem_id)
    if not problem:
        raise HTTPException(
            status_code=404,
            detail=f"Problem '{body.problem_id}' not found.",
        )

    # Build initial CoachState for the graph invocation.
    # gemini_api_key is threaded through state so critic_node can build
    # a fresh genai.Client per request — no server-side client singleton.
    initial_state = {
        "selected_topic":    body.topic,
        "current_problem":   problem,
        "user_code":         body.code,
        "evaluation_result": {},
        "history":           body.history,
        "critic_feedback":   "",
        "mastery_score":     body.mastery_score,
        "session_log":       body.session_log,
        "gemini_api_key":    getattr(body, "gemini_api_key", None),
    }

    run_state = eval_graph.invoke(
        initial_state,
        config={"recursion_limit": 10},
    )

    return ReviewCodeResponse(
        problem_id=body.problem_id,
        evaluation=_to_eval_result(run_state["evaluation_result"]),
        critic_feedback=run_state.get("critic_feedback", ""),
        updated_history=run_state.get("history", body.history),
        updated_mastery_score=run_state.get("mastery_score", body.mastery_score),
        updated_session_log=run_state.get("session_log", body.session_log),
    )
