"""
backend/services/executor.py
=============================
Sandboxed Python code execution engine.

Extracted verbatim from the original app.py evaluation logic.
Zero dependency on Streamlit, FastAPI, or LangGraph — pure stdlib only.
This means it can be unit-tested in complete isolation.

Security note
-------------
exec() runs arbitrary user code. For a production deployment you would
wrap this in a subprocess with resource limits (e.g. via RestrictedPython
or a nsjail sandbox). For a local/capstone app this is acceptable.
"""

from __future__ import annotations

import re
import traceback
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from typing import Any


# ── Core execution ────────────────────────────────────────────────────────────

def _run_test_case(user_code: str, test_case: dict, function_sig: str) -> dict:
    """
    Execute user code against a single test case in an isolated namespace.

    Returns a result dict: {passed, input, expected, actual, error}.
    """
    fn_name_match = re.search(r"def\s+(\w+)\s*\(", function_sig)
    if not fn_name_match:
        return {
            "passed": False,
            "error": "Could not parse function signature.",
            "actual": None,
            "input": test_case.get("input"),
            "expected": test_case.get("expected"),
        }

    expected_fn_name = fn_name_match.group(1)
    inputs   = test_case["input"]
    expected = test_case["expected"]

    namespace: dict[str, Any] = {}
    stdout_buf = StringIO()
    stderr_buf = StringIO()

    try:
        compiled = compile(user_code, "<user_code>", "exec")
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            exec(compiled, namespace)  # noqa: S102

        # Resolve function name — try declared name first, then scan user code
        if expected_fn_name in namespace:
            fn_name = expected_fn_name
        else:
            user_defined = re.findall(r"^def\s+(\w+)\s*\(", user_code, re.MULTILINE)
            fn_name = next(
                (name for name in user_defined if name in namespace),
                expected_fn_name,
            )

        if fn_name not in namespace:
            found = [k for k, v in namespace.items() if callable(v) and not k.startswith("_")]
            hint  = f" Found: {found}" if found else " No functions were defined."
            return {
                "passed":   False,
                "input":    inputs,
                "expected": expected,
                "actual":   None,
                "error": (
                    f"Function `{expected_fn_name}` not found in your submission.{hint}\n"
                    "Make sure your function name matches the signature exactly."
                ),
            }

        fn = namespace[fn_name]
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            actual = fn(**inputs)

        return {
            "passed":   _flexible_equal(actual, expected),
            "input":    inputs,
            "expected": expected,
            "actual":   actual,
            "error":    None,
        }

    except Exception:  # noqa: BLE001
        return {
            "passed":   False,
            "input":    inputs,
            "expected": expected,
            "actual":   None,
            "error":    traceback.format_exc(limit=4),
        }


def _flexible_equal(actual: Any, expected: Any) -> bool:
    """
    Tolerant equality for common algorithmic problem patterns:
      - list[list]  → sort inner lists before comparing (group anagram style)
      - list        → sort both (two_sum / anagram style)
      - primitive   → standard ==
    """
    if isinstance(expected, list) and isinstance(actual, list):
        if expected and isinstance(expected[0], list):
            return (
                sorted(sorted(x) for x in actual)
                == sorted(sorted(x) for x in expected)
            )
        return sorted(actual) == sorted(expected)
    return actual == expected


# ── Public API ────────────────────────────────────────────────────────────────

def evaluate_code(user_code: str, problem: dict) -> dict:
    """
    Run all test cases for a problem and return a summary dict.

    Return shape:
    {
        passed:       bool,
        passed_count: int,
        total:        int,
        summary:      str,
        results:      list[{passed, input, expected, actual, error}],
    }
    """
    test_cases = problem.get("test_cases", [])
    fn_sig     = problem.get("function_signature", "")

    if not user_code.strip():
        return {
            "passed":       False,
            "passed_count": 0,
            "total":        len(test_cases),
            "summary":      "⚠️ No code submitted.",
            "results":      [],
        }

    results      = [_run_test_case(user_code, tc, fn_sig) for tc in test_cases]
    passed_count = sum(1 for r in results if r["passed"])
    total        = len(results)
    all_passed   = passed_count == total

    return {
        "passed":       all_passed,
        "passed_count": passed_count,
        "total":        total,
        "summary":      (
            f"✅ All {total} test cases passed."
            if all_passed
            else f"❌ {passed_count}/{total} test cases passed."
        ),
        "results": results,
    }
