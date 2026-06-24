"""
AI LeetCode & Python Mastery Coach
====================================
A stateful, agentic coaching application built with LangGraph + Streamlit.

Architecture:
  - LangGraph StateGraph drives the adaptive learning loop
  - Four nodes: select_problem → evaluate_code → mastery | critic
  - Conditional routing based on evaluation outcome
  - Local JSON data vault for problem storage
  - Streamlit frontend with session state persistence
  - Google Gemini 2.5 Flash powers the Critic Agent (via google-genai SDK)
  - streamlit-code-editor provides a VS Code-style Monaco editor
"""

from __future__ import annotations

import json
import random
import re
import traceback
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, TypedDict, Optional

import streamlit as st
from google import genai
from code_editor import code_editor
from langgraph.graph import END, StateGraph

# ─────────────────────────────────────────────
# 0.  CONSTANTS & SETUP
# ─────────────────────────────────────────────

PROBLEMS_PATH = "problems.json"
TOPICS = ["Strings", "Lists", "Dictionaries", "Arrays"]
MAX_HISTORY = 6  # max problems to show in sidebar history

# ─────────────────────────────────────────────
# 1.  LANGGRAPH STATE
# ─────────────────────────────────────────────

class CoachState(TypedDict):
    """
    Single source of truth for the LangGraph coaching session.
    All nodes read from and write to this state dict.
    """
    selected_topic: str
    current_problem: dict           # full problem object from JSON vault
    user_code: str                  # raw code string submitted by user
    evaluation_result: dict         # {passed: bool, results: list, summary: str}
    history: dict                   # {passed: [ids], failed: [ids]}
    critic_feedback: str            # markdown critique from critic_node
    mastery_score: dict             # {topic: int} 0-100 per topic
    session_log: list[dict]         # ordered list of session events for the UI


# ─────────────────────────────────────────────
# 2.  DATA VAULT
# ─────────────────────────────────────────────

@st.cache_data
def load_problems() -> dict[str, list[dict]]:
    """Load and cache the local JSON problem vault."""
    with open(PROBLEMS_PATH, "r") as f:
        return json.load(f)


def get_problem_by_id(problem_id: str) -> Optional[dict]:
    """Look up a single problem across all topics."""
    problems = load_problems()
    for topic_problems in problems.values():
        for p in topic_problems:
            if p["id"] == problem_id:
                return p
    return None


# ─────────────────────────────────────────────
# 3.  CODE EVALUATION ENGINE
# ─────────────────────────────────────────────

def _run_test_case(user_code: str, test_case: dict, function_sig: str) -> dict:
    """
    Execute user code against a single test case in a controlled namespace.

    Returns a result dict with: passed, input, expected, actual, error.
    """
    # Extract the expected function name from the problem's function signature.
    # e.g. "def is_palindrome(s: str) -> bool:"  ->  "is_palindrome"
    fn_name_match = re.search(r"def\s+(\w+)\s*\(", function_sig)
    if not fn_name_match:
        return {"passed": False, "error": "Could not parse function signature", "actual": None}

    expected_fn_name = fn_name_match.group(1)
    inputs = test_case["input"]
    expected = test_case["expected"]

    namespace: dict[str, Any] = {}
    stdout_capture = StringIO()
    stderr_capture = StringIO()

    try:
        # Compile and exec user code into isolated namespace
        compiled = compile(user_code, "<user_code>", "exec")
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(compiled, namespace)  # noqa: S102

        # Resolve the callable to invoke.
        # Primary:  use the name declared in the problem's function_signature.
        # Fallback: if that name is not in the namespace (e.g. user renamed it),
        #           scan the submitted code for any def names actually present in
        #           the namespace, so the runner still works and the error is accurate.
        if expected_fn_name in namespace:
            fn_name = expected_fn_name
        else:
            user_defined = re.findall(r"^def\s+(\w+)\s*\(", user_code, re.MULTILINE)
            fn_name = next(
                (name for name in user_defined if name in namespace),
                expected_fn_name,  # fallback keeps original name for the error message
            )

        if fn_name not in namespace:
            found = [k for k, v in namespace.items() if callable(v) and not k.startswith("_")]
            hint = f" Found: {found}" if found else " No functions were defined."
            return {
                "passed": False,
                "error": (
                    f"Function `{expected_fn_name}` not found in your submission.{hint}\n"
                    "Make sure your function name matches the signature exactly."
                ),
                "actual": None,
            }

        fn = namespace[fn_name]

        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            actual = fn(**inputs)

        # Normalise list-based answers for order-independent problems
        passed = _flexible_equal(actual, expected)

        return {
            "passed": passed,
            "input": inputs,
            "expected": expected,
            "actual": actual,
            "error": None,
        }

    except Exception:  # noqa: BLE001
        return {
            "passed": False,
            "input": inputs,
            "expected": expected,
            "actual": None,
            "error": traceback.format_exc(limit=4),
        }


def _flexible_equal(actual: Any, expected: Any) -> bool:
    """
    Tolerant equality check:
     - Lists of lists → sorted inner lists before comparing (group anagram style)
     - Plain lists    → element-wise comparison after sorting where order doesn't matter
     - Primitives     → standard equality
    """
    if isinstance(expected, list) and isinstance(actual, list):
        if expected and isinstance(expected[0], list):
            return sorted(sorted(x) for x in actual) == sorted(sorted(x) for x in expected)
        # For two_sum style problems, sort both
        return sorted(actual) == sorted(expected)
    return actual == expected


def evaluate_code_programmatically(user_code: str, problem: dict) -> dict:
    """
    Run all hidden test cases and return an evaluation summary dict.
    """
    test_cases = problem.get("test_cases", [])
    fn_sig = problem.get("function_signature", "")
    results = []

    for tc in test_cases:
        result = _run_test_case(user_code, tc, fn_sig)
        results.append(result)

    passed_count = sum(1 for r in results if r["passed"])
    total = len(results)
    all_passed = passed_count == total

    return {
        "passed": all_passed,
        "passed_count": passed_count,
        "total": total,
        "results": results,
        "summary": f"{'✅ All' if all_passed else f'❌ {passed_count}/{total}'} test cases passed.",
    }


# ─────────────────────────────────────────────
# 4.  LANGGRAPH NODES
# ─────────────────────────────────────────────

def select_problem_node(state: CoachState) -> CoachState:
    """
    Spaced-repetition aware problem selector.
    Priority: failed problems first → new unseen problems → random from topic.
    """
    problems = load_problems()
    topic = state["selected_topic"]
    topic_problems = problems.get(topic, [])

    if not topic_problems:
        state["current_problem"] = {}
        return state

    history = state.get("history", {"passed": [], "failed": []})
    failed_ids = set(history.get("failed", []))
    passed_ids = set(history.get("passed", []))

    # 1. Retry a failed problem (spaced repetition)
    failed_in_topic = [p for p in topic_problems if p["id"] in failed_ids]
    if failed_in_topic:
        chosen = random.choice(failed_in_topic)
        state["current_problem"] = chosen
        return state

    # 2. Pick a new unseen problem
    unseen = [p for p in topic_problems if p["id"] not in passed_ids and p["id"] not in failed_ids]
    if unseen:
        chosen = random.choice(unseen)
        state["current_problem"] = chosen
        return state

    # 3. Fallback: random problem from topic (all seen/passed)
    chosen = random.choice(topic_problems)
    state["current_problem"] = chosen
    return state


def evaluate_code_node(state: CoachState) -> CoachState:
    """
    Programmatically evaluate user code against all hidden test cases.
    Stores results in state["evaluation_result"].
    """
    user_code = state.get("user_code", "")
    problem = state.get("current_problem", {})

    if not user_code.strip():
        state["evaluation_result"] = {
            "passed": False,
            "passed_count": 0,
            "total": 0,
            "results": [],
            "summary": "⚠️ No code submitted.",
        }
        return state

    eval_result = evaluate_code_programmatically(user_code, problem)
    state["evaluation_result"] = eval_result
    return state


def mastery_node(state: CoachState) -> CoachState:
    """
    Update mastery scores and history on a successful submission.
    Mastery score increases by 10 per unique passed problem (cap 100).
    """
    problem = state.get("current_problem", {})
    pid = problem.get("id", "")
    topic = state.get("selected_topic", "")

    history = state.get("history", {"passed": [], "failed": []})
    mastery = state.get("mastery_score", {t: 0 for t in TOPICS})

    # Move from failed → passed if it was a retry
    if pid in history.get("failed", []):
        history["failed"] = [x for x in history["failed"] if x != pid]

    # Record as passed
    if pid not in history.get("passed", []):
        history["passed"].append(pid)
        mastery[topic] = min(100, mastery.get(topic, 0) + 10)

    state["history"] = history
    state["mastery_score"] = mastery
    state["critic_feedback"] = ""  # clear any previous critique

    log_entry = {
        "type": "success",
        "problem_id": pid,
        "problem_title": problem.get("title", ""),
        "topic": topic,
        "summary": state["evaluation_result"].get("summary", ""),
        "mastery": mastery.get(topic, 0),
    }
    state["session_log"] = state.get("session_log", []) + [log_entry]
    return state


def critic_node(state: CoachState) -> CoachState:
    """
    LLM-powered deep code critique for failed submissions.
    Uses Google Gemini 2.5 Flash (picked up from st.session_state).
    """
    problem = state.get("current_problem", {})
    user_code = state.get("user_code", "")
    eval_result = state.get("evaluation_result", {})
    pid = problem.get("id", "")
    topic = state.get("selected_topic", "")

    # Record as failed in history
    history = state.get("history", {"passed": [], "failed": []})
    if pid and pid not in history.get("failed", []) and pid not in history.get("passed", []):
        history["failed"].append(pid)
    state["history"] = history

    # Build a compact failure report for the LLM
    failed_cases = [r for r in eval_result.get("results", []) if not r["passed"]]
    failure_details = "\n".join(
        f"  • Input: {r.get('input')} | Expected: {r.get('expected')} | Got: {r.get('actual')}"
        + (f"\n    Error: {r.get('error', '')[:300]}" if r.get("error") else "")
        for r in failed_cases[:3]  # limit to first 3 failures
    )

    system_prompt = """You are an elite Python & algorithms coach with the precision of a senior staff engineer.
Your role is to write a deeply analytical code critique that:
1. Identifies the ROOT CAUSE of the logical failure — not surface symptoms
2. Names specific anti-patterns or algorithmic misunderstandings (e.g., "Off-by-one in sliding window boundary", "Mutating iterable during iteration")
3. Explains WHY the correct approach works at a conceptual level
4. Provides a minimal, targeted code fix or pseudo-code outline — NOT the full solution
5. Ends with one concrete follow-up exercise to reinforce the concept

Format your response in clean Markdown with sections:
## 🔍 Root Cause Analysis
## ⚠️ Anti-Pattern Identified
## 💡 Conceptual Fix
## 🧪 Targeted Fix (pseudo-code / outline only)
## 🎯 Reinforcement Exercise

Be surgical and precise. No platitudes. Maximum depth in minimum words."""

    user_prompt = f"""Problem: **{problem.get('title', 'Unknown')}** ({topic})

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

Write a deep, precise critique."""

    try:
        client = st.session_state.get("google_client")
        if not client:
            state["critic_feedback"] = "⚠️ Google AI client not initialised. Please set your API key in the sidebar."
        else:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_prompt,
                config={"system_instruction": system_prompt},
            )
            state["critic_feedback"] = response.text
    except Exception as e:  # noqa: BLE001
        state["critic_feedback"] = f"⚠️ Critic agent error: {e}"

    log_entry = {
        "type": "failure",
        "problem_id": pid,
        "problem_title": problem.get("title", ""),
        "topic": topic,
        "summary": eval_result.get("summary", ""),
        "critic_preview": state["critic_feedback"][:80] + "…",
    }
    state["session_log"] = state.get("session_log", []) + [log_entry]
    return state


# ─────────────────────────────────────────────
# 5.  ROUTING LOGIC
# ─────────────────────────────────────────────

def route_after_evaluation(state: CoachState) -> str:
    """
    Conditional edge: route to mastery_node on pass, critic_node on fail.
    """
    if state.get("evaluation_result", {}).get("passed", False):
        return "mastery_node"
    return "critic_node"


# ─────────────────────────────────────────────
# 6.  GRAPH ASSEMBLY
# ─────────────────────────────────────────────

def build_graph() -> tuple[Any, Any]:
    """
    Build and return TWO compiled LangGraph StateGraphs:

      selection_graph  — entry: select_problem_node
                         Used ONLY when loading the next problem.
                         Flow: select_problem → evaluate → mastery | critic

      eval_graph       — entry: evaluate_code_node
                         Used ONLY when submitting user code.
                         Skips select_problem_node entirely so the current_problem
                         already in state is never replaced by a stale or random pick.
                         Flow: evaluate → mastery | critic

    Keeping them separate eliminates the bug where graph.invoke() during a submit
    would re-run select_problem_node and swap in a different problem's test cases
    before evaluate_code_node ran.
    """
    # ── Shared node factory (each graph needs its own StateGraph instance) ──

    def _make_eval_subgraph(include_select: bool) -> Any:
        g = StateGraph(CoachState)
        if include_select:
            g.add_node("select_problem_node", select_problem_node)
        g.add_node("evaluate_code_node", evaluate_code_node)
        g.add_node("mastery_node", mastery_node)
        g.add_node("critic_node", critic_node)

        if include_select:
            g.set_entry_point("select_problem_node")
            g.add_edge("select_problem_node", "evaluate_code_node")
        else:
            g.set_entry_point("evaluate_code_node")

        g.add_conditional_edges(
            "evaluate_code_node",
            route_after_evaluation,
            {
                "mastery_node": "mastery_node",
                "critic_node": "critic_node",
            },
        )
        g.add_edge("mastery_node", END)
        g.add_edge("critic_node", END)
        return g.compile()

    selection_graph = _make_eval_subgraph(include_select=True)
    eval_graph      = _make_eval_subgraph(include_select=False)
    return selection_graph, eval_graph


# ─────────────────────────────────────────────
# 7.  SESSION STATE HELPERS
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# 7.  SESSION STATE HELPERS
# ─────────────────────────────────────────────

def init_session_state():
    """
    Initialise all Streamlit session state keys on first load.

    Loop-prevention design
    ----------------------
    code_input        : The *committed* code string. Written ONLY on explicit button
                        events (submit / reset / load problem). The editor is always
                        seeded from this value; it is never overwritten on passive
                        re-renders, which breaks the blur->rerun->blur cycle.
    editor_event_id   : code_editor re-sends the last btnClick response payload on
                        every Streamlit re-render. Storing the last-seen response id
                        lets us skip duplicates and only act on genuinely new clicks.
    last_run_output   : Written once after the graph completes; read-only afterward
                        until a new submission fires.
    """
    defaults: dict[str, Any] = {
        "coach_state": {
            "selected_topic": "Strings",
            "current_problem": {},
            "user_code": "",
            "evaluation_result": {},
            "history": {"passed": [], "failed": []},
            "critic_feedback": "",
            "mastery_score": {t: 0 for t in TOPICS},
            "session_log": [],
        },
        # Build both graphs in one call so the StateGraph is only compiled once.
        # selection_graph is used for loading problems; eval_graph for submissions.
        **dict(zip(("graph", "eval_graph"), build_graph())),
        "google_client": None,
        "api_key_set": False,
        "code_input": "",
        "problem_loaded": False,
        "last_run_output": None,
        "editor_event_id": None,   # dedup guard — see note above
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ─────────────────────────────────────────────
# 8.  UI HELPERS
# ─────────────────────────────────────────────

def difficulty_badge(difficulty: str) -> str:
    colours = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}
    return colours.get(difficulty, "⚪") + f" **{difficulty}**"


def type_badge(problem_type: str) -> str:
    if problem_type == "syntax":
        return "🔤 *Syntax Mastery*"
    return "⚙️ *Algorithmic*"


def render_test_results(eval_result: dict):
    """Render test case pass/fail table in the console area."""
    results = eval_result.get("results", [])
    if not results:
        return

    st.markdown("##### Test Case Results")
    for i, r in enumerate(results, 1):
        icon = "✅" if r["passed"] else "❌"
        with st.expander(f"{icon} Case {i} — Input: `{r.get('input')}`", expanded=not r["passed"]):
            col1, col2 = st.columns(2)
            col1.markdown(f"**Expected:** `{r.get('expected')}`")
            col2.markdown(f"**Got:** `{r.get('actual')}`")
            if r.get("error"):
                st.code(r["error"], language="python")


def render_mastery_bar(topic: str, score: int):
    """Compact mastery progress bar."""
    filled = int(score / 10)
    bar = "█" * filled + "░" * (10 - filled)
    st.caption(f"`{bar}` {score}/100")


def _do_load_problem(cs: dict) -> None:
    """
    Run select_problem_node, seed code_input from the new problem's signature,
    and clear stale console state.  Does NOT call st.rerun() — callers decide.
    """
    tmp: CoachState = {
        **cs,
        "user_code": "",
        "evaluation_result": {},
        "critic_feedback": "",
    }
    new_state = select_problem_node(tmp)
    st.session_state.coach_state.update(new_state)
    sig = st.session_state.coach_state.get("current_problem", {}).get(
        "function_signature", ""
    )
    st.session_state.code_input = sig + "\n    pass\n"
    st.session_state.last_run_output = None
    st.session_state.coach_state["evaluation_result"] = {}
    st.session_state.coach_state["critic_feedback"] = ""
    st.session_state.problem_loaded = True


# ─────────────────────────────────────────────
# 9.  MAIN APP
# ─────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="AI Mastery Coach",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Custom CSS ─────────────────────────────
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600&display=swap');

      html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

      section[data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid #21262d;
      }
      section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
      section[data-testid="stSidebar"] .stRadio label { font-size: 0.9rem; }

      .console-box {
        background: #0d1117;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        line-height: 1.6;
        color: #c9d1d9;
        min-height: 120px;
      }

      .critic-box {
        background: #161b22;
        border-left: 4px solid #f85149;
        border-radius: 0 8px 8px 0;
        padding: 1.2rem 1.4rem;
        margin: 0.8rem 0;
        color: #c9d1d9;
      }

      .success-box {
        background: #0d2818;
        border-left: 4px solid #3fb950;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.4rem;
        color: #3fb950;
      }

      .problem-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 1.4rem;
        margin-bottom: 1rem;
      }

      .topic-pill {
        display: inline-block;
        background: #1f6feb22;
        border: 1px solid #1f6feb;
        color: #58a6ff;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
      }

      .stButton > button {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        border-radius: 6px;
      }

      #MainMenu, footer { visibility: hidden; }
      h1 { font-size: 1.6rem !important; }
      h2 { font-size: 1.25rem !important; border-bottom: 1px solid #21262d; padding-bottom: 0.3rem; }
    </style>
    """, unsafe_allow_html=True)

    init_session_state()

    # ── SIDEBAR ────────────────────────────────
    with st.sidebar:
        st.markdown("## 🧠 Mastery Coach")
        st.markdown("---")

        st.markdown("**Google AI Studio API Key**")
        api_key_input = st.text_input(
            "API Key",
            type="password",
            placeholder="AIza...",
            key="api_key_input",
            label_visibility="collapsed",
        )
        if api_key_input and not st.session_state.api_key_set:
            st.session_state.google_client = genai.Client(api_key=api_key_input)
            st.session_state.api_key_set = True
            st.success("✓ API key saved")

        st.markdown("---")

        st.markdown("**Select Topic**")
        selected_topic = st.radio(
            "Topic",
            TOPICS,
            label_visibility="collapsed",
            key="topic_radio",
        )

        # Topic change: clear problem + editor state in-place.
        # No st.rerun() needed — the cleared state renders correctly on this pass.
        current_topic = st.session_state.coach_state.get("selected_topic")
        if selected_topic != current_topic:
            st.session_state.coach_state["selected_topic"] = selected_topic
            st.session_state.coach_state["current_problem"] = {}
            st.session_state.coach_state["evaluation_result"] = {}
            st.session_state.coach_state["critic_feedback"] = ""
            st.session_state.problem_loaded = False
            st.session_state.code_input = ""
            st.session_state.last_run_output = None
            st.session_state.editor_event_id = None

        st.markdown("---")

        st.markdown("**Topic Mastery**")
        mastery = st.session_state.coach_state.get("mastery_score", {t: 0 for t in TOPICS})
        for t in TOPICS:
            score = mastery.get(t, 0)
            icon = "🌟" if score >= 80 else ("📈" if score >= 40 else "📚")
            st.markdown(f"{icon} **{t}**")
            render_mastery_bar(t, score)

        st.markdown("---")

        session_log = st.session_state.coach_state.get("session_log", [])
        if session_log:
            st.markdown("**Session History**")
            for entry in reversed(session_log[-MAX_HISTORY:]):
                icon = "✅" if entry["type"] == "success" else "❌"
                st.caption(f"{icon} {entry.get('problem_title', '?')} · {entry.get('topic', '')}")

        st.markdown("---")
        st.caption("Built with LangGraph · Gemini 2.5 Flash · Streamlit")

    # ── MAIN CONTENT ───────────────────────────
    st.markdown("# 🧠 AI LeetCode & Python Mastery Coach")

    cs = st.session_state.coach_state
    topic = cs["selected_topic"]

    col_problem, col_console = st.columns([1.05, 1], gap="large")

    # ── LEFT COLUMN: Problem Panel ─────────────
    with col_problem:
        st.markdown("## Problem")

        # Load Next Problem — the ONE place we call st.rerun(), because we need
        # the editor component to re-mount with a fresh `code` seed value.
        if st.button("⟳  Load Next Problem", type="primary", use_container_width=True):
            _do_load_problem(cs)
            st.session_state.editor_event_id = None
            st.rerun()

        problem = cs.get("current_problem", {})

        if not problem:
            st.info("👆 Click **Load Next Problem** to begin your session.")
        else:
            # ── Problem card ──────────────────────────────────────────
            st.markdown(f"""
            <div class="problem-card">
              <div class="topic-pill">{topic}</div>
              <h3 style="margin: 0.4rem 0 0.2rem; color: #e6edf3;">{problem.get('title', '')}</h3>
              <p style="margin: 0; color: #8b949e; font-size:0.85rem;">
                {difficulty_badge(problem.get('difficulty', ''))} &nbsp;·&nbsp; {type_badge(problem.get('type', ''))}
              </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Description**")
            st.markdown(problem.get("description", ""))

            if problem.get("examples"):
                st.markdown("**Examples**")
                for ex in problem["examples"]:
                    st.markdown(f"- **Input:** `{ex['input']}` → **Output:** `{ex['output']}`")

            if problem.get("constraints"):
                with st.expander("📋 Constraints"):
                    for c in problem["constraints"]:
                        st.markdown(f"- {c}")

            if problem.get("hints"):
                with st.expander("💡 Hints (tap to reveal)"):
                    for h in problem["hints"]:
                        st.markdown(f"- {h}")

            st.markdown("---")
            st.markdown("**Your Solution**")

            # ── VS Code-style editor ──────────────────────────────────
            #
            # INFINITE-LOOP ROOT CAUSE & FIX
            # --------------------------------
            # The bug: response_mode=["blur", "btnClick"] fires on EVERY focus-leave
            # event, including when Streamlit re-renders any other widget on the page.
            # Because we were writing to st.session_state.code_input on every render
            # (not just on clicks), that write triggered another rerun, which re-rendered
            # the editor, which blurred, which wrote again — an infinite cycle.
            #
            # The fix has three parts:
            #
            # 1. response_mode=["btnClick"] ONLY.
            #    The editor now only sends a payload when the user explicitly clicks
            #    Submit or Reset. Passive re-renders return {} — nothing to act on.
            #
            # 2. Dedup guard via editor_event_id.
            #    code_editor re-sends the last btnClick payload on every subsequent
            #    Streamlit re-render (it's stored in the component's React state).
            #    We compare editor_response["id"] against st.session_state.editor_event_id
            #    and only process the event if the id is new.
            #
            # 3. No st.rerun() after graph execution.
            #    After running the graph we let the current render pass continue.
            #    The updated coach_state and last_run_output are already in session_state,
            #    so the console column below picks them up immediately with no extra rerun.

            editor_seed = st.session_state.code_input or (
                problem.get("function_signature", "") + "\n    pass\n"
            )

            editor_buttons = [
                {
                    "name": "▶  Submit",
                    "feather": "Play",
                    "primary": True,
                    "hasText": True,
                    "showWithIcon": True,
                    "commands": ["submit"],
                    "style": {
                        "bottom": "0.5rem",
                        "right": "0.5rem",
                        "backgroundColor": "#238636",
                        "color": "#ffffff",
                        "border": "none",
                        "borderRadius": "6px",
                        "padding": "6px 14px",
                        "fontWeight": "600",
                        "fontSize": "0.82rem",
                        "cursor": "pointer",
                    },
                },
                {
                    "name": "↺  Reset",
                    "feather": "RefreshCw",
                    "primary": False,
                    "hasText": True,
                    "showWithIcon": True,
                    "commands": ["reset"],
                    "style": {
                        "bottom": "0.5rem",
                        "right": "7.5rem",
                        "backgroundColor": "#21262d",
                        "color": "#c9d1d9",
                        "border": "1px solid #30363d",
                        "borderRadius": "6px",
                        "padding": "6px 14px",
                        "fontWeight": "600",
                        "fontSize": "0.82rem",
                        "cursor": "pointer",
                    },
                },
            ]

            editor_css = """
                .ace-vscode .ace_gutter { background: #1e1e1e; color: #858585; }
                .ace-vscode .ace_scroller { background: #1e1e1e; }
                .ace-vscode .ace_cursor { color: #aeafad; }
                .ace-vscode .ace_marker-layer .ace_selection { background: #264f78; }
                .ace-vscode .ace_keyword { color: #569cd6; }
                .ace-vscode .ace_string { color: #ce9178; }
                .ace-vscode .ace_comment { color: #6a9955; font-style: italic; }
                .ace-vscode .ace_numeric { color: #b5cea8; }
                .ace-vscode .ace_identifier { color: #9cdcfe; }
                .ace-vscode .ace_paren { color: #ffd700; }
                .ace-vscode .ace_function { color: #dcdcaa; }
                .ace-vscode .ace_active-line { background: #2a2d2e; }
                .ace-vscode .ace_gutter-active-line { background: #1e1e1e; color: #c6c6c6; }
                body { background: #1e1e1e !important; }
            """

            editor_response = code_editor(
                code=editor_seed,
                lang="python",
                theme="vs-dark",
                height=[18, 28],
                buttons=editor_buttons,
                focus=True,
                key="vscode_editor",
                options={
                    "wrap": False,
                    "tabSize": 4,
                    "useSoftTabs": True,
                    "enableBasicAutocompletion": True,
                    "enableLiveAutocompletion": True,
                    "enableSnippets": True,
                    "showGutter": True,
                    "showLineNumbers": True,
                    "highlightActiveLine": True,
                    "showPrintMargin": False,
                    "fontSize": 14,
                    "fontFamily": "'JetBrains Mono', 'Fira Code', monospace",
                },
                # css=editor_css,
                response_mode=["btnClick"],  # ← CRITICAL: no "blur"
            )

            # ── Process editor events ─────────────────────────────────
            # editor_response is {} on passive renders (no button clicked).
            # On a real click it is {"type": "submit"|"reset", "text": "...", "id": "..."}
            # We only act when:
            #   (a) the response is non-empty and has a "type" field, AND
            #   (b) the response "id" differs from the last one we already handled.
            # This prevents the re-sent-on-rerender payload from firing twice.

            event_type = editor_response.get("type", "") if editor_response else ""
            event_id   = editor_response.get("id",   "") if editor_response else ""

            is_new_event = (
                event_type in ("submit", "reset")
                and event_id
                and event_id != st.session_state.editor_event_id
            )

            if is_new_event:
                # Mark this event as handled before doing any work,
                # so even if something below causes a rerun the dedup guard holds.
                st.session_state.editor_event_id = event_id

                if event_type == "reset":
                    # Restore the clean function signature and wipe console output.
                    # We call st.rerun() so the editor re-mounts with the reset seed.
                    st.session_state.code_input = (
                        problem.get("function_signature", "") + "\n    pass\n"
                    )
                    st.session_state.last_run_output = None
                    cs["evaluation_result"] = {}
                    cs["critic_feedback"] = ""
                    st.rerun()

                elif event_type == "submit":
                    submitted_code = editor_response.get("text", "").strip()

                    if not submitted_code:
                        st.warning("⚠️ Editor is empty — write your solution before submitting.")
                    elif not st.session_state.api_key_set:
                        st.warning("⚠️ Please enter your Google AI Studio API key in the sidebar first.")
                    else:
                        # Commit the code to our stable store.
                        st.session_state.code_input = submitted_code

                        with st.spinner("Running test cases…"):
                            # Use eval_graph (entry: evaluate_code_node) so that
                            # select_problem_node is never called during a submission.
                            # This guarantees the test cases always match the problem
                            # currently displayed in the UI, not a randomly re-selected one.
                            run_state = st.session_state.eval_graph.invoke(
                                {**cs, "user_code": submitted_code},
                                config={"recursion_limit": 10},
                            )

                        # Write results to session state.
                        # NO st.rerun() here — the console column below reads from
                        # the same session_state dict and renders on this same pass.
                        st.session_state.coach_state.update(run_state)
                        st.session_state.last_run_output = run_state

    # ── RIGHT COLUMN: Console / Feedback ───────
    # This column is drawn in the same render pass as the editor event handler above,
    # so updated results are visible immediately without an extra rerun.
    with col_console:
        st.markdown("## Console")

        output     = st.session_state.last_run_output
        eval_result    = cs.get("evaluation_result", {})
        critic_feedback = cs.get("critic_feedback", "")

        if not output and not eval_result:
            st.markdown("""
            <div class="console-box">
              <span style="color:#6e7681;">// Output will appear here after you submit your solution.<br>
              // Test cases run programmatically against your code.<br>
              // Failures trigger the Critic Agent for deep analysis.</span>
            </div>
            """, unsafe_allow_html=True)

        if eval_result:
            summary = eval_result.get("summary", "")

            if eval_result.get("passed"):
                mastery_score = cs.get("mastery_score", {}).get(topic, 0)
                st.markdown(f"""
                <div class="success-box">
                  <strong>🎉 All Test Cases Passed!</strong><br>
                  {summary}<br>
                  <span style="font-size:0.85rem; opacity:0.8;">
                    {topic} Mastery: {mastery_score}/100
                  </span>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                <div style="background:#2d1117; border-left:4px solid #f85149;
                            border-radius:0 8px 8px 0; padding:0.8rem 1.2rem;
                            margin-bottom:0.8rem; color:#f85149;">
                  <strong>❌ {summary}</strong>
                </div>
                """, unsafe_allow_html=True)

            render_test_results(eval_result)

            if critic_feedback:
                st.markdown("---")
                st.markdown("### 🤖 Critic Agent Review")
                st.markdown(f"""
                <div class="critic-box">
                {critic_feedback}
                </div>
                """, unsafe_allow_html=True)

        # Load-next shortcut — shown only after a successful submission.
        # st.rerun() here is correct: we need the editor to re-mount with the
        # new problem's signature as its seed value.
        if eval_result.get("passed") and output:
            st.markdown("---")
            if st.button("→ Load Next Problem", key="next_btn", use_container_width=True):
                _do_load_problem(cs)
                st.session_state.editor_event_id = None
                st.rerun()


if __name__ == "__main__":
    main()