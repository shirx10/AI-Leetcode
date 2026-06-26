# AI LeetCode Coach — FastAPI + Streamlit Architecture

## Directory Structure

```
coach_app/
│
├── backend/
│   ├── main.py                  # FastAPI app, CORS, router registration
│   ├── models.py                # All Pydantic request/response models
│   │
│   ├── routers/
│   │   ├── problems.py          # GET /api/problems, GET /api/problems/{id},
│   │   │                        # POST /api/problems/next
│   │   ├── execution.py         # POST /api/run-tests  (fast path, no LLM)
│   │   └── agent.py             # POST /api/review-code (full LangGraph loop)
│   │
│   ├── services/
│   │   ├── data_vault.py        # File I/O, problem lookups, boilerplate gen
│   │   └── executor.py          # Code execution engine (exec + test runner)
│   │
│   └── graph/
│       ├── state.py             # CoachState TypedDict
│       ├── nodes.py             # All LangGraph node functions
│       └── builder.py           # Compiled graph singletons (eval + selection)
│
├── frontend/
│   └── app.py                   # Streamlit UI — pure presentation, no logic
│
├── problems.json                # Problem vault (unchanged)
└── requirements.txt
```

## Design Principles

### 1. Strict Layer Separation
Each layer has one job:
- **`services/`** — pure Python, zero framework imports (no FastAPI, no Streamlit)
- **`graph/`** — LangGraph only; reads from services, no HTTP
- **`routers/`** — HTTP boundary; validates input, calls services/graph, formats output
- **`frontend/app.py`** — renders UI and makes HTTP requests; stores zero business logic

### 2. Stateless Backend
The API keeps no per-user session.  Every request from Streamlit sends the
full coaching context (`history`, `mastery_score`, `session_log`), and the
response returns the updated versions of those fields.  The Streamlit app
stores them in `st.session_state` and echoes them back on the next call.

This means you can replace Streamlit with any other frontend (React, CLI, etc.)
without touching a single line of backend code.

### 3. Two-Graph Design (preserved)
`eval_graph` (entry: `evaluate_code_node`) is used for submissions.
`selection_graph` (entry: `select_problem_node`) is used for problem loading.
This prevents the original bug where invoking the graph during a submission
would re-run `select_problem_node` and swap in a different problem's test cases.

### 4. Gemini API Key Flow
The API key is **never stored on the server**.
- User enters it in the Streamlit sidebar → stored in `st.session_state.gemini_api_key`
- Sent in the `POST /api/review-code` request body as `gemini_api_key`
- Threaded through `CoachState["gemini_api_key"]` to `critic_node`
- `critic_node` creates a fresh `genai.Client(api_key=...)` per request

## Running the App

```bash
# Install dependencies
pip install -r requirements.txt

# Terminal 1: start the API server
uvicorn backend.main:app --reload --port 8000

# Terminal 2: start the Streamlit frontend
streamlit run frontend/app.py
```

Open http://localhost:8501 for the UI.
Open http://localhost:8000/docs for the interactive Swagger API explorer.

## API Endpoints

| Method | Path                    | Purpose                                      |
|--------|-------------------------|----------------------------------------------|
| GET    | `/health`               | Liveness check                               |
| GET    | `/api/problems`         | Full problem vault                           |
| GET    | `/api/problems/{id}`    | Single problem detail                        |
| POST   | `/api/problems/next`    | Spaced-repetition next problem selection     |
| POST   | `/api/run-tests`        | Execute code, return test results (no LLM)   |
| POST   | `/api/review-code`      | Full loop: evaluate + mastery + Critic Agent |

## Key Differences from the Monolith

| Concern               | Old (app.py)                          | New (FastAPI)                              |
|-----------------------|---------------------------------------|--------------------------------------------|
| Business logic        | Entangled with Streamlit widgets      | Isolated in `services/` and `graph/`       |
| Session state         | `st.session_state` dict               | Client-side; echoed in every API request   |
| LLM client            | Stored in `st.session_state`          | Created per-request from body key          |
| Graph compilation     | Inside `init_session_state()`         | Module-level singleton in `builder.py`     |
| Editor reset bug      | Required `_bump_editor_key()` hack    | Solved cleanly: reset → API clears state   |
| Test isolation        | Impossible (Streamlit required)       | `executor.py` and `nodes.py` are pure Python, fully unit-testable |
