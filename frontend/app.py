"""
frontend/app.py
================
Streamlit frontend for the AI LeetCode Coach.

Architecture after migration
-----------------------------
This file contains ZERO business logic.  Every action calls the FastAPI
backend and renders the JSON response.  Session state now only stores
UI-level concerns (editor key, last response) and the user's coaching
progress (history, mastery) — the latter is sent back to the API on each
request so the backend remains fully stateless.

Run alongside the backend:
    # Terminal 1
    uvicorn backend.main:app --reload --port 8000

    # Terminal 2
    streamlit run frontend/app.py
"""

from __future__ import annotations

import requests
import streamlit as st
from code_editor import code_editor

API_BASE = "http://localhost:8000/api"
TOPICS   = ["Strings", "Lists", "Dictionaries", "Arrays"]


# ─────────────────────────────────────────────────────────────
# API client helpers
# ─────────────────────────────────────────────────────────────

def _post(endpoint: str, payload: dict) -> dict:
    """POST to the backend and return the parsed JSON, or an error dict."""
    try:
        r = requests.post(f"{API_BASE}/{endpoint}", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        detail = e.response.json().get("detail", str(e)) if e.response else str(e)
        return {"_error": detail}
    except Exception as e:
        return {"_error": str(e)}


def _get(endpoint: str) -> dict:
    try:
        r = requests.get(f"{API_BASE}/{endpoint}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"_error": str(e)}


def api_next_problem(topic: str) -> dict:
    return _post("problems/next", {
        "topic":         topic,
        "history":       st.session_state.history,
        "mastery_score": st.session_state.mastery_score,
    })


def api_run_tests(problem_id: str, code: str) -> dict:
    return _post("run-tests", {"problem_id": problem_id, "code": code})


def api_review_code(problem_id: str, code: str, topic: str) -> dict:
    return _post("review-code", {
        "problem_id":    problem_id,
        "code":          code,
        "topic":         topic,
        "history":       st.session_state.history,
        "mastery_score": st.session_state.mastery_score,
        "session_log":   st.session_state.session_log,
        "gemini_api_key": st.session_state.gemini_api_key,
    })


# ─────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        # User coaching progress — sent to API on every submission
        "history":       {"passed": [], "failed": []},
        "mastery_score": {t: 0 for t in TOPICS},
        "session_log":   [],
        # Current problem
        "current_problem": None,
        "active_topic":    TOPICS[0],
        # Editor state
        "code_input":      "",
        "editor_key":      "editor_0",
        "editor_event_id": None,
        # Last API response (for console rendering)
        "last_eval":       None,
        "last_feedback":   "",
        # Auth
        "gemini_api_key":  "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _bump_editor_key():
    """Force code_editor to remount as a clean instance."""
    prefix, _, n = st.session_state.editor_key.rpartition("_")
    st.session_state.editor_key = f"{prefix}_{int(n) + 1}"


def _load_problem(topic: str):
    """Call the API, store the returned problem, reset editor and console."""
    resp = api_next_problem(topic)
    if "_error" in resp:
        st.error(f"❌ {resp['_error']}")
        return

    problem = resp.get("problem")
    if not problem:
        st.error("No problem returned from API.")
        return

    st.session_state.current_problem = problem
    st.session_state.code_input      = problem["boilerplate"]
    st.session_state.last_eval        = None
    st.session_state.last_feedback    = ""
    st.session_state.editor_event_id  = None
    _bump_editor_key()


# ─────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────

def difficulty_badge(d: str) -> str:
    return {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}.get(d, "⚪") + f" **{d}**"


def type_badge(t: str) -> str:
    return "🔤 *Syntax Mastery*" if t == "syntax" else "⚙️ *Algorithmic*"


def render_mastery_bar(score: int):
    filled = int(score / 10)
    bar    = "█" * filled + "░" * (10 - filled)
    st.caption(f"`{bar}` {score}/100")


def render_test_results(eval_result: dict):
    results = eval_result.get("results", [])
    if not results:
        return
    st.markdown("##### Test Case Results")
    for i, r in enumerate(results, 1):
        icon = "✅" if r["passed"] else "❌"
        with st.expander(f"{icon} Case {i} — Input: `{r.get('input')}`", expanded=not r["passed"]):
            c1, c2 = st.columns(2)
            c1.markdown(f"**Expected:** `{r.get('expected')}`")
            c2.markdown(f"**Got:** `{r.get('actual')}`")
            if r.get("error"):
                st.code(r["error"], language="python")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="AI Mastery Coach",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600&display=swap');
      html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
      section[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #21262d; }
      section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
      .console-box { background:#0d1117; border:1px solid #21262d; border-radius:8px;
                     padding:1.2rem 1.4rem; font-family:'JetBrains Mono',monospace;
                     font-size:0.82rem; line-height:1.6; color:#c9d1d9; min-height:120px; }
      .critic-box { background:#161b22; border-left:4px solid #f85149;
                    border-radius:0 8px 8px 0; padding:1.2rem 1.4rem; margin:0.8rem 0; color:#c9d1d9; }
      .success-box { background:#0d2818; border-left:4px solid #3fb950;
                     border-radius:0 8px 8px 0; padding:1rem 1.4rem; color:#3fb950; }
      .problem-card { background:#161b22; border:1px solid #21262d;
                      border-radius:10px; padding:1.4rem; margin-bottom:1rem; }
      .topic-pill { display:inline-block; background:#1f6feb22; border:1px solid #1f6feb;
                    color:#58a6ff; border-radius:20px; padding:2px 12px;
                    font-size:0.78rem; font-weight:600; margin-bottom:0.4rem; }
      .stButton > button { font-family:'Inter',sans-serif; font-weight:600; border-radius:6px; }
      #MainMenu, footer { visibility:hidden; }
    </style>
    """, unsafe_allow_html=True)

    init_state()

    # ── Sidebar ───────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🧠 Mastery Coach")
        st.markdown("---")

        # API key input
        key_input = st.text_input(
            "Google AI Studio API Key",
            type="password",
            placeholder="AIza...",
            value=st.session_state.gemini_api_key,
            label_visibility="visible",
        )
        if key_input != st.session_state.gemini_api_key:
            st.session_state.gemini_api_key = key_input
            if key_input:
                st.success("✓ API key saved")

        st.markdown("---")

        # Topic selector
        st.markdown("**Select Topic**")
        selected_topic = st.radio(
            "Topic",
            TOPICS,
            index=TOPICS.index(st.session_state.active_topic),
            label_visibility="collapsed",
        )

        # Topic change → clear problem and editor
        if selected_topic != st.session_state.active_topic:
            st.session_state.active_topic     = selected_topic
            st.session_state.current_problem   = None
            st.session_state.code_input        = ""
            st.session_state.last_eval         = None
            st.session_state.last_feedback     = ""
            st.session_state.editor_event_id   = None
            _bump_editor_key()

        st.markdown("---")

        # Mastery bars
        st.markdown("**Topic Mastery**")
        for t in TOPICS:
            score = st.session_state.mastery_score.get(t, 0)
            icon  = "🌟" if score >= 80 else ("📈" if score >= 40 else "📚")
            st.markdown(f"{icon} **{t}**")
            render_mastery_bar(score)

        st.markdown("---")

        # Session history
        if st.session_state.session_log:
            st.markdown("**Session History**")
            for entry in reversed(st.session_state.session_log[-6:]):
                icon = "✅" if entry.get("type") == "success" else "❌"
                st.caption(f"{icon} {entry.get('problem_title','?')} · {entry.get('topic','')}")

        st.markdown("---")
        st.caption("Built with LangGraph · Gemini 2.5 Flash · FastAPI · Streamlit")

    # ── Main content ──────────────────────────────────────────
    st.markdown("# 🧠 AI LeetCode & Python Mastery Coach")

    topic   = st.session_state.active_topic
    problem = st.session_state.current_problem

    col_problem, col_console = st.columns([1.05, 1], gap="large")

    # ── LEFT: Problem panel ───────────────────────────────────
    with col_problem:
        st.markdown("## Problem")

        if st.button("⟳  Load Next Problem", type="primary", use_container_width=True):
            with st.spinner("Selecting problem…"):
                _load_problem(topic)
            st.rerun()

        problem = st.session_state.current_problem   # re-read after possible update

        if not problem:
            st.info("👆 Click **Load Next Problem** to begin your session.")
        else:
            # Problem card
            st.markdown(f"""
            <div class="problem-card">
              <div class="topic-pill">{topic}</div>
              <h3 style="margin:0.4rem 0 0.2rem; color:#e6edf3;">{problem['title']}</h3>
              <p style="margin:0; color:#8b949e; font-size:0.85rem;">
                {difficulty_badge(problem.get('difficulty',''))}
                &nbsp;·&nbsp;
                {type_badge(problem.get('type',''))}
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

            # ── Editor ────────────────────────────────────────
            editor_seed = st.session_state.code_input or problem.get("boilerplate", "")

            editor_buttons = [
                {
                    "name": "▶  Submit",
                    "feather": "Play",
                    "primary": True,
                    "hasText": True,
                    "showWithIcon": True,
                    "commands": ["submit"],
                    "style": {
                        "bottom": "0.5rem", "right": "0.5rem",
                        "backgroundColor": "#238636", "color": "#ffffff",
                        "border": "none", "borderRadius": "6px",
                        "padding": "6px 14px", "fontWeight": "600",
                        "fontSize": "0.82rem", "cursor": "pointer",
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
                        "bottom": "0.5rem", "right": "7.5rem",
                        "backgroundColor": "#21262d", "color": "#c9d1d9",
                        "border": "1px solid #30363d", "borderRadius": "6px",
                        "padding": "6px 14px", "fontWeight": "600",
                        "fontSize": "0.82rem", "cursor": "pointer",
                    },
                },
            ]

            editor_response = code_editor(
                code=editor_seed,
                lang="python",
                theme="vs-dark",
                height=[18, 28],
                buttons=editor_buttons,
                focus=True,
                key=st.session_state.editor_key,
                options={
                    "wrap": False, "tabSize": 4, "useSoftTabs": True,
                    "enableBasicAutocompletion": True,
                    "enableLiveAutocompletion": True,
                    "enableSnippets": True,
                    "showGutter": True, "showLineNumbers": True,
                    "highlightActiveLine": True, "showPrintMargin": False,
                    "fontSize": 14,
                    "fontFamily": "'JetBrains Mono', 'Fira Code', monospace",
                },
                response_mode=["btnClick"],
            )

            # ── Handle editor events ──────────────────────────
            event_type = editor_response.get("type", "") if editor_response else ""
            event_id   = editor_response.get("id",   "") if editor_response else ""

            is_new_event = (
                event_type in ("submit", "reset")
                and event_id
                and event_id != st.session_state.editor_event_id
            )

            if is_new_event:
                st.session_state.editor_event_id = event_id

                if event_type == "reset":
                    # ── Reset: restore boilerplate, clear console, remount editor
                    st.session_state.code_input  = problem.get("boilerplate", "")
                    st.session_state.last_eval   = None
                    st.session_state.last_feedback = ""
                    _bump_editor_key()
                    st.rerun()

                elif event_type == "submit":
                    submitted_code = editor_response.get("text", "").strip()

                    if not submitted_code:
                        st.warning("⚠️ Editor is empty.")
                    elif not st.session_state.gemini_api_key:
                        st.warning("⚠️ Please enter your Google AI Studio API key in the sidebar.")
                    else:
                        st.session_state.code_input = submitted_code

                        with st.spinner("Running tests & generating feedback…"):
                            resp = api_review_code(
                                problem_id=problem["id"],
                                code=submitted_code,
                                topic=topic,
                            )

                        if "_error" in resp:
                            st.error(f"API error: {resp['_error']}")
                        else:
                            # Persist updated coaching state from the response
                            st.session_state.last_eval      = resp["evaluation"]
                            st.session_state.last_feedback  = resp.get("critic_feedback", "")
                            st.session_state.history        = resp["updated_history"]
                            st.session_state.mastery_score  = resp["updated_mastery_score"]
                            st.session_state.session_log    = resp["updated_session_log"]

    # ── RIGHT: Console ────────────────────────────────────────
    with col_console:
        st.markdown("## Console")

        eval_result     = st.session_state.last_eval
        critic_feedback = st.session_state.last_feedback

        if not eval_result:
            st.markdown("""
            <div class="console-box">
              <span style="color:#6e7681;">
                // Output will appear here after you submit.<br>
                // Test cases run against your code on the backend.<br>
                // Failures trigger the Critic Agent for deep analysis.
              </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            if eval_result.get("passed"):
                score = st.session_state.mastery_score.get(topic, 0)
                st.markdown(f"""
                <div class="success-box">
                  <strong>🎉 All Test Cases Passed!</strong><br>
                  {eval_result['summary']}<br>
                  <span style="font-size:0.85rem;opacity:0.8;">
                    {topic} Mastery: {score}/100
                  </span>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                <div style="background:#2d1117;border-left:4px solid #f85149;
                            border-radius:0 8px 8px 0;padding:0.8rem 1.2rem;
                            margin-bottom:0.8rem;color:#f85149;">
                  <strong>❌ {eval_result['summary']}</strong>
                </div>
                """, unsafe_allow_html=True)

            render_test_results(eval_result)

            if critic_feedback:
                st.markdown("---")
                st.markdown("### 🤖 Critic Agent Review")
                st.markdown(f'<div class="critic-box">{critic_feedback}</div>',
                            unsafe_allow_html=True)

        # Load next shortcut after a pass
        if eval_result and eval_result.get("passed"):
            st.markdown("---")
            if st.button("→ Load Next Problem", key="next_btn", use_container_width=True):
                with st.spinner("Selecting next problem…"):
                    _load_problem(topic)
                st.rerun()


if __name__ == "__main__":
    main()
