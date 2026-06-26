"""
backend/graph/nodes.py
=======================
All LangGraph node functions.

Key change from the original app.py:
  - critic_node no longer reads from st.session_state to get the Gemini client.
    Instead, the API key is threaded through CoachState["gemini_api_key"] so
    the graph is completely stateless and can run in any async context.
"""

from __future__ import annotations

import random
from typing import Any

from google import genai

from backend.graph.state import CoachState, TOPICS
from backend.services.data_vault import get_problems_by_topic
from backend.services.executor import evaluate_code


# ─────────────────────────────────────────────────────────────
# Node: select_problem
# ─────────────────────────────────────────────────────────────

def select_problem_node(state: CoachState) -> CoachState:
    """
    Spaced-repetition aware problem selector.
    Priority: failed problems → unseen problems → random fallback.
    """
    topic          = state["selected_topic"]
    topic_problems = get_problems_by_topic(topic)

    if not topic_problems:
        state["current_problem"] = {}
        return state

    history    = state.get("history", {"passed": [], "failed": []})
    failed_ids = set(history.get("failed", []))
    passed_ids = set(history.get("passed", []))

    # 1. Retry failed
    failed_in_topic = [p for p in topic_problems if p["id"] in failed_ids]
    if failed_in_topic:
        state["current_problem"] = random.choice(failed_in_topic)
        return state

    # 2. Pick unseen
    unseen = [
        p for p in topic_problems
        if p["id"] not in passed_ids and p["id"] not in failed_ids
    ]
    if unseen:
        state["current_problem"] = random.choice(unseen)
        return state

    # 3. Fallback
    state["current_problem"] = random.choice(topic_problems)
    return state


# ─────────────────────────────────────────────────────────────
# Node: evaluate_code
# ─────────────────────────────────────────────────────────────

def evaluate_code_node(state: CoachState) -> CoachState:
    """Run all test cases and store the evaluation result in state."""
    state["evaluation_result"] = evaluate_code(
        state.get("user_code", ""),
        state.get("current_problem", {}),
    )
    return state


# ─────────────────────────────────────────────────────────────
# Node: mastery
# ─────────────────────────────────────────────────────────────

def mastery_node(state: CoachState) -> CoachState:
    """Update mastery scores and history on a successful submission."""
    problem = state.get("current_problem", {})
    pid     = problem.get("id", "")
    topic   = state.get("selected_topic", "")

    history = state.get("history", {"passed": [], "failed": []})
    mastery = state.get("mastery_score", {t: 0 for t in TOPICS})

    # Promote from failed → passed
    if pid in history.get("failed", []):
        history["failed"] = [x for x in history["failed"] if x != pid]

    if pid not in history.get("passed", []):
        history["passed"].append(pid)
        mastery[topic] = min(100, mastery.get(topic, 0) + 10)

    state["history"]      = history
    state["mastery_score"] = mastery
    state["critic_feedback"] = ""

    state["session_log"] = state.get("session_log", []) + [{
        "type":          "success",
        "problem_id":    pid,
        "problem_title": problem.get("title", ""),
        "topic":         topic,
        "summary":       state["evaluation_result"].get("summary", ""),
        "mastery":       mastery.get(topic, 0),
    }]
    return state


# ─────────────────────────────────────────────────────────────
# Node: critic
# ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an elite Python & algorithms coach with the precision of a senior staff engineer.
Your role is to write a deeply analytical code critique that:
1. Identifies the ROOT CAUSE of the logical failure — not surface symptoms
2. Names specific anti-patterns or algorithmic misunderstandings
3. Explains WHY the correct approach works at a conceptual level
4. Provides a minimal, targeted code fix or pseudo-code outline — NOT the full solution
5. Ends with one concrete follow-up exercise to reinforce the concept

Format your response in clean Markdown with sections:
## 🔍 Root Cause Analysis
## ⚠️ Anti-Pattern Identified
## 💡 Conceptual Fix
## 🧪 Targeted Fix (pseudo-code / outline only)
## 🎯 Reinforcement Exercise

Be surgical and precise. No platitudes. Maximum depth in minimum words.\
"""


def critic_node(state: CoachState) -> CoachState:
    """
    LLM-powered deep code critique for failed submissions.

    Reads gemini_api_key from state (injected by the API endpoint at
    request time) instead of from st.session_state.
    """
    problem     = state.get("current_problem", {})
    user_code   = state.get("user_code", "")
    eval_result = state.get("evaluation_result", {})
    pid         = problem.get("id", "")
    topic       = state.get("selected_topic", "")
    api_key     = state.get("gemini_api_key", "")

    # Track as failed
    history = state.get("history", {"passed": [], "failed": []})
    if pid and pid not in history.get("failed", []) and pid not in history.get("passed", []):
        history["failed"].append(pid)
    state["history"] = history

    # Build failure summary
    failed_cases = [r for r in eval_result.get("results", []) if not r["passed"]]
    failure_details = "\n".join(
        f"  • Input: {r.get('input')} | Expected: {r.get('expected')} | Got: {r.get('actual')}"
        + (f"\n    Error: {r.get('error', '')[:300]}" if r.get("error") else "")
        for r in failed_cases[:3]
    )

    user_prompt = f"""\
Problem: **{problem.get('title', 'Unknown')}** ({topic})

Function signature:
```python
{problem.get('function_signature', '')}
```

User's submitted code:
```python
{user_code}
```

Test case failures ({len(failed_cases)} failed out of {eval_result.get('total', 0)}):
{failure_details}

Write a deep, precise critique.\
"""

    try:
        if not api_key:
            state["critic_feedback"] = "⚠️ No Gemini API key provided."
        else:
            client   = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_prompt,
                config={"system_instruction": _SYSTEM_PROMPT},
            )
            state["critic_feedback"] = response.text
    except Exception as exc:  # noqa: BLE001
        state["critic_feedback"] = f"⚠️ Critic agent error: {exc}"

    state["session_log"] = state.get("session_log", []) + [{
        "type":           "failure",
        "problem_id":     pid,
        "problem_title":  problem.get("title", ""),
        "topic":          topic,
        "summary":        eval_result.get("summary", ""),
        "critic_preview": state["critic_feedback"][:80] + "…",
    }]
    return state


# ─────────────────────────────────────────────────────────────
# Routing
# ─────────────────────────────────────────────────────────────

def route_after_evaluation(state: CoachState) -> str:
    if state.get("evaluation_result", {}).get("passed", False):
        return "mastery_node"
    return "critic_node"
