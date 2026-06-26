"""
backend/graph/state.py
=======================
CoachState TypedDict — the single source of truth for the LangGraph graph.

Kept in its own module so every node file can import it without
creating circular dependencies.
"""

from __future__ import annotations
from typing import TypedDict


TOPICS = ["Strings", "Lists", "Dictionaries", "Arrays"]


class CoachState(TypedDict):
    selected_topic:    str
    current_problem:   dict        # full problem object from vault
    user_code:         str
    evaluation_result: dict        # {passed, passed_count, total, summary, results}
    history:           dict        # {passed: [ids], failed: [ids]}
    critic_feedback:   str
    mastery_score:     dict        # {topic: int}  0-100
    session_log:       list[dict]
    # Injected at request time — not persisted in the graph itself
    gemini_api_key:    str
