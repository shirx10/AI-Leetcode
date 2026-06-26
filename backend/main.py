"""
backend/main.py
================
FastAPI application entry point.

Run with:
    uvicorn backend.main:app --reload --port 8000

Interactive docs available at:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.models import HealthResponse
from backend.routers import agent, execution, problems

app = FastAPI(
    title="AI LeetCode Coach API",
    description=(
        "FastAPI backend for the LangGraph-powered Python coaching app. "
        "Provides code execution, spaced-repetition problem selection, "
        "and Gemini-powered critic feedback."
    ),
    version="1.0.0",
)

# ── CORS — allow the Streamlit dev server (port 8501) to call us ──────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",   # Streamlit default
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(problems.router)
app.include_router(execution.router)
app.include_router(agent.router)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health():
    return HealthResponse(status="ok")
