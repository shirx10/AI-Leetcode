"""
backend/models.py
=================
All Pydantic models used across the API.

Design rule: every endpoint speaks strictly in these types.
The routers never construct raw dicts — they return model instances,
and FastAPI serialises them to JSON automatically.
"""

from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────
# Inbound request bodies
# ─────────────────────────────────────────────────────────────

class RunTestsRequest(BaseModel):
    """POST /api/run-tests"""
    problem_id: str = Field(..., description="Unique problem identifier from problems.json")
    code: str       = Field(..., description="Raw Python source code submitted by the user")


class ReviewCodeRequest(BaseModel):
    """POST /api/review-code"""
    problem_id:     str  = Field(..., description="Problem the user was solving")
    code:           str  = Field(..., description="User's submitted code")
    topic:          str  = Field(..., description="Active topic, e.g. 'Strings'")
    # Full session history so the LangGraph mastery/spaced-repetition nodes
    # can update state correctly without keeping server-side session.
    history:        dict = Field(default_factory=lambda: {"passed": [], "failed": []})
    mastery_score:  dict = Field(default_factory=dict)
    session_log:    list = Field(default_factory=list)


class NextProblemRequest(BaseModel):
    """POST /api/problems/next"""
    topic:         str  = Field(..., description="Topic to select from")
    history:       dict = Field(default_factory=lambda: {"passed": [], "failed": []})
    mastery_score: dict = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────
# Sub-shapes used inside responses
# ─────────────────────────────────────────────────────────────

class TestCaseResult(BaseModel):
    passed:   bool
    input:    Optional[Any]   = None
    expected: Optional[Any]   = None
    actual:   Optional[Any]   = None
    error:    Optional[str]   = None


class EvaluationResult(BaseModel):
    passed:       bool
    passed_count: int
    total:        int
    summary:      str
    results:      list[TestCaseResult] = Field(default_factory=list)


class ProblemSummary(BaseModel):
    """Lightweight view — used in list endpoints and next-problem selection."""
    id:                 str
    title:              str
    topic:              str
    difficulty:         str
    type:               str
    description:        str
    function_signature: str
    boilerplate:        str          # signature + "\n    pass\n"
    examples:           list[dict]   = Field(default_factory=list)
    constraints:        list[str]    = Field(default_factory=list)
    hints:              list[str]    = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# Outbound response bodies
# ─────────────────────────────────────────────────────────────

class RunTestsResponse(BaseModel):
    """POST /api/run-tests"""
    problem_id:        str
    evaluation:        EvaluationResult


class ReviewCodeResponse(BaseModel):
    """POST /api/review-code"""
    problem_id:        str
    evaluation:        EvaluationResult
    critic_feedback:   str
    # Updated session state — caller stores this client-side and sends it back
    # on the next request.  No server-side session needed.
    updated_history:       dict
    updated_mastery_score: dict
    updated_session_log:   list


class NextProblemResponse(BaseModel):
    """POST /api/problems/next"""
    problem: ProblemSummary


class ProblemsListResponse(BaseModel):
    """GET /api/problems"""
    topics:   list[str]
    problems: dict[str, list[ProblemSummary]]


class HealthResponse(BaseModel):
    status: str = "ok"
