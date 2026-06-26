"""
backend/routers/problems.py
============================
Problem management endpoints.

GET  /api/problems              — full vault (all topics + problems)
GET  /api/problems/{problem_id} — single problem detail
POST /api/problems/next         — spaced-repetition next problem selector
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.models import (
    NextProblemRequest,
    NextProblemResponse,
    ProblemsListResponse,
    ProblemSummary,
)
from backend.services.data_vault import (
    enrich_problem,
    get_all_problems,
    get_problem_by_id,
    TOPICS,
)
from backend.graph.builder import selection_graph

router = APIRouter(prefix="/api/problems", tags=["problems"])


def _to_summary(problem: dict) -> ProblemSummary:
    """Convert a raw vault dict to a ProblemSummary model."""
    enriched = enrich_problem(problem)
    return ProblemSummary(
        id=enriched["id"],
        title=enriched["title"],
        topic=enriched.get("topic", ""),
        difficulty=enriched.get("difficulty", ""),
        type=enriched.get("type", ""),
        description=enriched.get("description", ""),
        function_signature=enriched.get("function_signature", ""),
        boilerplate=enriched["boilerplate"],
        examples=enriched.get("examples", []),
        constraints=enriched.get("constraints", []),
        hints=enriched.get("hints", []),
    )


@router.get("", response_model=ProblemsListResponse)
async def list_problems():
    """Return the complete problem vault grouped by topic."""
    all_problems = get_all_problems()
    return ProblemsListResponse(
        topics=TOPICS,
        problems={
            topic: [_to_summary(p) for p in problems]
            for topic, problems in all_problems.items()
        },
    )


@router.get("/{problem_id}", response_model=ProblemSummary)
async def get_problem(problem_id: str):
    """Return a single problem by ID."""
    problem = get_problem_by_id(problem_id)
    if not problem:
        raise HTTPException(status_code=404, detail=f"Problem '{problem_id}' not found.")
    return _to_summary(problem)


@router.post("/next", response_model=NextProblemResponse)
async def next_problem(body: NextProblemRequest):
    """
    Use the spaced-repetition selector to pick the next problem for the user.

    The caller passes its current history + mastery state; the server runs
    select_problem_node and returns the chosen problem.  No server-side
    session is required.
    """
    initial_state = {
        "selected_topic":    body.topic,
        "current_problem":   {},
        "user_code":         "",
        "evaluation_result": {},
        "history":           body.history,
        "critic_feedback":   "",
        "mastery_score":     body.mastery_score,
        "session_log":       [],
        "gemini_api_key":    "",   # not needed for selection
    }

    # Run only the selection node — we don't evaluate here
    from backend.graph.nodes import select_problem_node
    new_state = select_problem_node(initial_state)  # type: ignore[arg-type]

    problem = new_state.get("current_problem")
    if not problem:
        raise HTTPException(
            status_code=404,
            detail=f"No problems available for topic '{body.topic}'.",
        )

    return NextProblemResponse(problem=_to_summary(problem))
