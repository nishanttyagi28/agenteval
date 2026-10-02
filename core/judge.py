"""Provider-neutral correctness judge for open-ended cases.

The default chain does not require the Agentic Data Analyst sibling repository:

1. ``AGENTEVAL_JUDGE_PROVIDER`` when set (``offline``, ``openai``, ``groq``,
   ``anthropic``, ``legacy``).
2. An API key already in the environment (OpenAI, then Groq, then Anthropic).
3. The legacy sibling client, only when that repository is actually present
   or the caller passed ``agent_repo`` (``agenteval calibrate``).
4. The deterministic ``offline`` judge.

``legacy`` keeps the pre-0.5 Groq client that lives in the sibling repo.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

_PROVIDER_ENV = "AGENTEVAL_JUDGE_PROVIDER"


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


def select_judge_provider(
    explicit: str | None = None,
    *,
    agent_repo: str | Path | None = None,
) -> str:
    """Resolve which judge implementation should score this case."""
    chosen = (explicit if explicit is not None else os.environ.get(_PROVIDER_ENV) or "").strip()
    if chosen:
        return chosen.lower()
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return "openai"
    if os.environ.get("GROQ_API_KEY", "").strip():
        return "groq"
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return "anthropic"
    if agent_repo is not None:
        return "legacy"
    from agenteval.core.config import AgentDependencyNotFound, resolve_agent_repo

    try:
        resolve_agent_repo()
    except AgentDependencyNotFound:
        return "offline"
    return "legacy"


def judge_correctness(
    prompt: str,
    final_answer: str,
    ground_truth: Any,
    *,
    temperature: float = 0.0,
    max_tokens: int = 256,
    agent_repo: str | Path | None = None,
    provider: str | None = None,
) -> tuple[bool, str]:
    """Judge one (prompt, answer) pair against ``ground_truth``.

    ``agent_repo`` is only used by the ``legacy`` provider. ``provider`` overrides
    ``AGENTEVAL_JUDGE_PROVIDER`` for this call.
    """
    chosen = select_judge_provider(provider, agent_repo=agent_repo)
    if chosen == "legacy":
        return _legacy_sibling_judge(
            prompt,
            final_answer,
            ground_truth,
            temperature=temperature,
            max_tokens=max_tokens,
            agent_repo=agent_repo,
        )
    from agenteval.core.judge_providers import judge_with_provider

    return judge_with_provider(
        chosen,
        prompt,
        final_answer,
        ground_truth,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _legacy_sibling_judge(
    prompt: str,
    final_answer: str,
    ground_truth: Any,
    *,
    temperature: float,
    max_tokens: int,
    agent_repo: str | Path | None,
) -> tuple[bool, str]:
    """Pre-0.5 path: import ``agents.llm_client`` from the sibling repository."""
    from agenteval.core.config import resolve_agent_repo

    resolved = resolve_agent_repo(explicit=agent_repo)
    if str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))
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
