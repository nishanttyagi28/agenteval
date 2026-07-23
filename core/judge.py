"""Groq LLM judge used only for genuinely open-ended correctness cases."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from agenteval.core.config import resolve_agent_repo


def _parse_judge_response(text: str) -> tuple[bool | None, str]:
    if not text or not text.strip():
        return None, "empty judge response"
    raw = text.strip()
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
    agent_repo: str | Path | None = None,
) -> tuple[bool, str]:
    """Judge one (prompt, answer) pair against ``ground_truth``.

    ``agent_repo``, when given, is used instead of the default single-agent
    discovery in ``core.config.resolve_agent_repo`` -- this lets callers that
    already resolved a specific registered agent's sibling repository (e.g.
    ``agenteval calibrate --judge <name>``, via ``core.registry.
    resolve_agent_repository``) point the judge at that same repository's LLM
    client. Omitting it (the default) preserves the original single-agent
    behavior exactly.
    """
    agent_repo = resolve_agent_repo(explicit=agent_repo)
    if str(agent_repo) not in sys.path:
        sys.path.insert(0, str(agent_repo))
    from utils.env import load_project_env

    load_project_env()
    from agents.llm_client import chat_completion

    criteria = ground_truth if ground_truth is not None else "(no extra criteria)"
    if not isinstance(criteria, str):
        criteria = json.dumps(criteria, ensure_ascii=False)
    system = (
        "You are a strict evaluation judge for a data-analysis agent. "
        "Decide if the answer is acceptably correct. Require correct substance and no "
        "major invented facts. Respond with JSON only: "
        '{"pass": true|false, "reason": "one short line"}'
    )
    user = (
        f"User question:\n{prompt}\n\nCorrectness criteria:\n{criteria}\n\n"
        f"Agent answer:\n{final_answer or '(empty)'}\n"
    )
    response, error = chat_completion(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if error:
        return False, f"judge error: {error}"
    if response is None:
        return False, "judge error: empty response"
    passed, reason = _parse_judge_response(response)
    if passed is None:
        return False, reason
    return passed, reason[:300]
