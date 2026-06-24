# 🧠 AI LeetCode & Python Mastery Coach

A stateful, adaptive learning coach built with **LangGraph + Streamlit + Anthropic**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    LangGraph StateGraph                  │
│                                                         │
│   ┌──────────────────────┐                             │
│   │  select_problem_node  │ ← Spaced repetition:       │
│   │  (Entry Point)        │   failed first → new →     │
│   └──────────┬───────────┘   random fallback           │
│              │                                          │
│   ┌──────────▼───────────┐                             │
│   │  evaluate_code_node   │ ← Programmatic sandbox:    │
│   │                       │   exec() in isolated ns    │
│   └──────────┬───────────┘                             │
│              │ add_conditional_edges                    │
│         ┌────▼────┐                                     │
│         │ passed? │                                     │
│         └──┬───┬──┘                                     │
│         YES│   │NO                                      │
│   ┌────────▼┐ ┌▼──────────────┐                        │
│   │ mastery │ │  critic_node   │ ← LLM deep critique   │
│   │  _node  │ │  (Anthropic)   │   structured output   │
│   └────┬────┘ └──────┬─────────┘                       │
│        │             │                                  │
│       END           END                                 │
└─────────────────────────────────────────────────────────┘
```

## File Structure

```
.
├── app.py              # Main application (Streamlit + LangGraph)
├── problems.json       # Local data vault (16 problems across 4 topics)
├── requirements.txt    # Python dependencies
└── README.md
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

Then enter your **Anthropic API key** in the sidebar.

---

## Core Components

### `CoachState` (TypedDict)
```python
class CoachState(TypedDict):
    selected_topic: str
    current_problem: dict       # full problem object from JSON vault
    user_code: str              # raw code submitted by user
    evaluation_result: dict     # {passed, passed_count, total, results, summary}
    history: dict               # {passed: [ids], failed: [ids]}
    critic_feedback: str        # markdown critique from critic_node
    mastery_score: dict         # {topic: 0-100}
    session_log: list[dict]     # ordered event log for sidebar history
```

### Graph Nodes

| Node | Responsibility |
|------|---------------|
| `select_problem_node` | Spaced-repetition aware selector. Retries failed problems first. |
| `evaluate_code_node` | Runs `exec()` sandbox, checks all hidden test cases. |
| `mastery_node` | Updates mastery score (+10/unique pass, capped at 100). |
| `critic_node` | Calls Anthropic LLM with structured prompt for deep code critique. |

### Conditional Routing

```python
graph.add_conditional_edges(
    "evaluate_code_node",
    route_after_evaluation,          # returns "mastery_node" or "critic_node"
    {"mastery_node": "mastery_node", "critic_node": "critic_node"},
)
```

### `problems.json` Schema

```json
{
  "TopicName": [
    {
      "id": "STR-001",
      "title": "Problem Title",
      "difficulty": "Easy | Medium | Hard",
      "type": "algorithmic | syntax",
      "description": "Markdown problem statement",
      "examples": [{"input": "...", "output": "..."}],
      "constraints": ["..."],
      "function_signature": "def fn_name(args) -> return_type:",
      "test_cases": [
        {"input": {"arg1": value}, "expected": value}
      ],
      "hints": ["..."]
    }
  ]
}
```

---

## Extending the Vault

Add problems to `problems.json` following the schema above. The `function_signature` field is
critical — the evaluation engine parses it to locate the user's function by name.

## Adaptive Learning Logic

1. **First visit** → pick an unseen problem from the selected topic
2. **Failure** → add to `history.failed`, Critic Agent writes review
3. **Next session** → `select_problem_node` checks `history.failed` first (spaced repetition)
4. **Retry success** → remove from `failed`, add to `passed`, increment mastery score
5. **All problems seen** → shuffle and re-serve from the topic pool
