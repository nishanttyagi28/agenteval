"""LLM-as-judge (Groq) — used ONLY for correctness_type=llm_judge."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

# Ensure monorepo root is importable for agents.llm_client
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parse_judge_response(text: str) -> tuple[bool | None, str]:
    """Parse pass/fail + one-line reason from judge model output."""
    if not text or not text.strip():
        return None, "empty judge response"

    raw = text.strip()
    # Prefer JSON blob
    match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            passed = data.get("pass", data.get("passed", data.get("correct")))
            reason = str(data.get("reason") or data.get("explanation") or "").strip()
            if isinstance(passed, str):
                passed = passed.strip().lower() in ("true", "pass", "yes", "1")
            if isinstance(passed, bool):
                return passed, reason or ("pass" if passed else "fail")
        except json.JSONDecodeError:
            pass

    lower = raw.lower()
    if re.search(r"\bpass\b", lower) and not re.search(r"\bfail\b", lower):
        return True, raw.splitlines()[0][:240]
    if re.search(r"\bfail\b", lower):
        return False, raw.splitlines()[0][:240]
    return None, f"unparseable judge output: {raw[:200]}"


def judge_correctness(
    prompt: str,
    final_answer: str,
    ground_truth: Any,
    *,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> tuple[bool, str]:
    """
    Ask Groq whether ``final_answer`` is correct for ``prompt`` given criteria.

    Returns (passed, one_line_reason). On API failure returns (False, error reason).
    """
    from utils.env import load_project_env

    load_project_env()
    from agents.llm_client import chat_completion

    criteria = ground_truth if ground_truth is not None else "(no extra criteria)"
    if not isinstance(criteria, str):
        criteria = json.dumps(criteria, ensure_ascii=False)

    system = (
        "You are a strict evaluation judge for a data-analysis agent. "
        "Decide if the agent's answer is acceptably correct for the user question. "
        "Be fair: wording may differ; require correct substance and no major invented facts. "
        'Respond with JSON only: {"pass": true|false, "reason": "one short line"}'
    )
    user = (
        f"User question:\n{prompt}\n\n"
        f"Correctness criteria / ground truth notes:\n{criteria}\n\n"
        f"Agent answer:\n{final_answer or '(empty)'}\n"
    )

    response, err = chat_completion(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if err:
        return False, f"judge error: {err}"
    if response is None:
        return False, "judge error: empty response"

    passed, reason = _parse_judge_response(response)
    if passed is None:
        return False, reason
    return passed, reason[:300]
