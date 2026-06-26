"""
backend/services/data_vault.py
==============================
Pure data-access layer for the problem vault.

No Streamlit, no FastAPI — just file I/O and simple lookups.
The FastAPI app caches the loaded data at startup via a module-level
singleton so the file is only read once per process.
"""

from __future__ import annotations

import json
import functools
from pathlib import Path
from typing import Optional

PROBLEMS_PATH = Path(__file__).parent.parent.parent / "problems.json"
TOPICS = ["Strings", "Lists", "Dictionaries", "Arrays"]


# ── Module-level cache (loaded once at import time) ───────────────────────────

@functools.lru_cache(maxsize=1)
def load_problems() -> dict[str, list[dict]]:
    """
    Load the problems vault from disk.
    lru_cache(1) means the file is read exactly once per interpreter lifetime.
    Call load_problems.cache_clear() in tests to reset.
    """
    with open(PROBLEMS_PATH, "r") as f:
        return json.load(f)


# ── Lookup helpers ────────────────────────────────────────────────────────────

def get_all_problems() -> dict[str, list[dict]]:
    return load_problems()


def get_problems_by_topic(topic: str) -> list[dict]:
    return load_problems().get(topic, [])


def get_problem_by_id(problem_id: str) -> Optional[dict]:
    """Linear scan across all topics — vault is small so this is fine."""
    for topic_problems in load_problems().values():
        for p in topic_problems:
            if p["id"] == problem_id:
                return p
    return None


def make_boilerplate(problem: dict) -> str:
    """Return the starter code stub for a given problem."""
    sig = problem.get("function_signature", "def solution():")
    return sig + "\n    pass\n"


def enrich_problem(problem: dict) -> dict:
    """Add derived fields (boilerplate, topic) to a raw problem dict."""
    enriched = dict(problem)
    enriched.setdefault("boilerplate", make_boilerplate(problem))
    # Attach the topic string if the vault doesn't embed it per-problem
    if "topic" not in enriched:
        for topic, problems in load_problems().items():
            if any(p["id"] == problem["id"] for p in problems):
                enriched["topic"] = topic
                break
    return enriched
