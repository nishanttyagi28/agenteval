"""Provider-neutral judges for open-ended correctness.

``offline`` is deterministic and makes no network call. ``openai``, ``groq``,
and ``anthropic`` use the stdlib HTTP client. Set ``AGENTEVAL_JUDGE_PROVIDER``
to pick one. ``AGENTEVAL_JUDGE_BASE_URL`` points ``openai`` and ``groq`` at any
OpenAI-compatible endpoint (Azure, a proxy, a local server).
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

from agenteval.core.judge import _parse_judge_response

PROVIDERS = frozenset({"offline", "openai", "groq", "anthropic", "legacy"})

_DEFAULT_MODELS = {
    "openai": "gpt-4.1-mini",
    "groq": "llama-3.3-70b-versatile",
    "anthropic": "claude-3-5-haiku-20241022",
}

_SYSTEM = (
    "You are a strict evaluation judge. Decide if the agent answer is acceptably "
    "correct given the criteria. Require the right substance and no major invented "
    "facts. Respond with JSON only: "
    '{"pass": true|false, "reason": "one short line"}'
)


def judge_prompt(prompt: str, final_answer: str, ground_truth: Any) -> str:
    criteria = ground_truth if ground_truth is not None else "(no extra criteria)"
    if not isinstance(criteria, str):
        criteria = json.dumps(criteria, ensure_ascii=False)
    return (
        f"User question:\n{prompt}\n\nCorrectness criteria:\n{criteria}\n\n"
        f"Agent answer:\n{final_answer or '(empty)'}\n"
    )


def offline_judge(prompt: str, final_answer: str, ground_truth: Any) -> tuple[bool, str]:
    """Deterministic stand-in for an LLM judge. No network, no sibling repo."""
    answer = final_answer or ""
    if not answer.strip():
        return False, "offline judge: empty answer"
    if ground_truth is None:
        return False, "offline judge: no ground truth to check"

    if isinstance(ground_truth, dict):
        must = ground_truth.get("must_include") or ground_truth.get("contains") or []
        must_not = ground_truth.get("must_not_include") or []
        if isinstance(must, str):
            must = [must]
        if isinstance(must_not, str):
            must_not = [must_not]
        missing = [item for item in must if str(item).lower() not in answer.lower()]
        leaked = [item for item in must_not if str(item).lower() in answer.lower()]
        if missing or leaked:
            bits = []
            if missing:
                bits.append("missing " + ", ".join(str(item) for item in missing))
            if leaked:
                bits.append("forbidden " + ", ".join(str(item) for item in leaked))
            return False, "offline judge: " + "; ".join(bits)
        criteria = ground_truth.get("criteria") or ground_truth.get("text")
        if criteria is None and (must or must_not):
            return True, "offline judge: include and exclude rules satisfied"
        if criteria is None:
            return _text_criteria_match(answer, json.dumps(ground_truth, ensure_ascii=False))
        return _text_criteria_match(answer, str(criteria))

    if isinstance(ground_truth, list):
        text = " ".join(str(item) for item in ground_truth)
        return _text_criteria_match(answer, text)
    return _text_criteria_match(answer, str(ground_truth))


def _text_criteria_match(answer: str, criteria: str) -> tuple[bool, str]:
    text = criteria.strip()
    if not text or text == "(no extra criteria)":
        return False, "offline judge: empty criteria"
    if text.lower() in answer.lower():
        return True, "offline judge: criteria text present in the answer"
    tokens = re.findall(r"[A-Za-z0-9_]{4,}", text)
    if tokens and all(token.lower() in answer.lower() for token in tokens):
        return True, "offline judge: criteria terms present in the answer"
    return False, "offline judge: answer does not satisfy the criteria"


def judge_with_provider(
    provider: str,
    prompt: str,
    final_answer: str,
    ground_truth: Any,
    *,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> tuple[bool, str]:
    name = (provider or "").strip().lower()
    if name == "offline":
        return offline_judge(prompt, final_answer, ground_truth)
    if name in ("openai", "groq", "anthropic"):
        return _remote_judge(
            name,
            prompt,
            final_answer,
            ground_truth,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    return False, f"judge error: unknown provider {provider!r}"


def _model_for(provider: str) -> str:
    override = os.environ.get("AGENTEVAL_JUDGE_MODEL", "").strip()
    if override:
        return override
    return _DEFAULT_MODELS[provider]


def _timeout() -> float:
    raw = os.environ.get("AGENTEVAL_JUDGE_TIMEOUT", "30").strip()
    try:
        value = float(raw)
    except ValueError:
        return 30.0
    return value if value > 0 else 30.0


def _remote_judge(
    provider: str,
    prompt: str,
    final_answer: str,
    ground_truth: Any,
    *,
    temperature: float,
    max_tokens: int,
) -> tuple[bool, str]:
    user = judge_prompt(prompt, final_answer, ground_truth)
    try:
        if provider == "anthropic":
            text = _anthropic_complete(user, temperature=temperature, max_tokens=max_tokens)
        else:
            text = _openai_compatible_complete(
                provider, user, temperature=temperature, max_tokens=max_tokens
            )
    except Exception as exc:  # noqa: BLE001 — judge must not crash a suite
        return False, f"judge error: {type(exc).__name__}: {exc}"[:300]
    passed, reason = _parse_judge_response(text)
    if passed is None:
        return False, reason[:300]
    return passed, reason[:300]


def _openai_compatible_complete(
    provider: str, user: str, *, temperature: float, max_tokens: int
) -> str:
    key_env = "OPENAI_API_KEY" if provider == "openai" else "GROQ_API_KEY"
    api_key = os.environ.get(key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"{key_env} is not set")
    base = os.environ.get("AGENTEVAL_JUDGE_BASE_URL", "").strip().rstrip("/")
    if not base:
        base = (
            "https://api.openai.com/v1"
            if provider == "openai"
            else "https://api.groq.com/openai/v1"
        )
    url = f"{base}/chat/completions"
    body = {
        "model": _model_for(provider),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ],
    }
    data = _post_json(
        url,
        body,
        {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        return str(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("provider response missing message content") from exc


def _anthropic_complete(user: str, *, temperature: float, max_tokens: int) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    body = {
        "model": _model_for("anthropic"),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": _SYSTEM,
        "messages": [{"role": "user", "content": user}],
    }
    data = _post_json(
        "https://api.anthropic.com/v1/messages",
        body,
        {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
    )
    try:
        return str(data["content"][0]["text"])
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("provider response missing message content") from exc


def _post_json(url: str, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=_timeout()) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:180]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request failed: {exc.reason}") from exc
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("provider returned non-JSON") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("provider returned a non-object JSON body")
    return parsed
