"""
backend/graph/builder.py
=========================
Compiles the two LangGraph StateGraphs and caches them as module-level
singletons.  FastAPI imports `eval_graph` and `selection_graph` directly.

Two-graph design (preserved from original app.py):
  selection_graph  — entry: select_problem_node
                     Used by POST /api/problems/next
                     Flow: select → evaluate → mastery | critic

  eval_graph       — entry: evaluate_code_node
                     Used by POST /api/run-tests and POST /api/review-code
                     Skips select_problem_node so the current_problem in
                     state is never replaced during a code submission.
                     Flow: evaluate → mastery | critic
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from backend.graph.state import CoachState
from backend.graph.nodes import (
    critic_node,
    evaluate_code_node,
    mastery_node,
    route_after_evaluation,
    select_problem_node,
)


def _build(include_select: bool) -> Any:
    g = StateGraph(CoachState)

    if include_select:
        g.add_node("select_problem_node", select_problem_node)

    g.add_node("evaluate_code_node", evaluate_code_node)
    g.add_node("mastery_node",       mastery_node)
    g.add_node("critic_node",        critic_node)

    if include_select:
        g.set_entry_point("select_problem_node")
        g.add_edge("select_problem_node", "evaluate_code_node")
    else:
        g.set_entry_point("evaluate_code_node")

    g.add_conditional_edges(
        "evaluate_code_node",
        route_after_evaluation,
        {"mastery_node": "mastery_node", "critic_node": "critic_node"},
    )
    g.add_edge("mastery_node", END)
    g.add_edge("critic_node",  END)
    return g.compile()


# ── Module-level singletons (compiled once at import time) ────────────────────
selection_graph = _build(include_select=True)
eval_graph      = _build(include_select=False)
