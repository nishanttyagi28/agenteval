"""Provider-neutral judge: offline by default, HTTP when a key is configured."""

from __future__ import annotations

import io
import json

import pytest

from agenteval.core.config import AgentDependencyNotFound
from agenteval.core.judge import judge_correctness, select_judge_provider
from agenteval.core.judge_providers import offline_judge


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    for name in (
        "AGENTEVAL_JUDGE_PROVIDER",
        "OPENAI_API_KEY",
        "GROQ_API_KEY",
        "ANTHROPIC_API_KEY",
        "AGENTEVAL_JUDGE_MODEL",
        "AGENTEVAL_JUDGE_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_offline_judge_passes_when_criteria_text_is_in_the_answer():
    passed, reason = offline_judge(
        "Refund order 4821",
        "Refund issued for order 4821",
        "Refund issued",
    )
    assert passed is True
    assert "criteria" in reason


def test_offline_judge_fails_an_empty_or_unrelated_answer():
    passed, reason = offline_judge("Refund order 4821", "", "Refund issued")
    assert passed is False
    assert "empty" in reason
    passed, reason = offline_judge("Refund order 4821", "Cancelled the order", "Refund issued")
    assert passed is False


def test_offline_judge_honors_include_and_exclude_rules():
    passed, reason = offline_judge(
        "status?",
        "refund issued, no extra charge",
        {"must_include": ["refund issued"], "must_not_include": ["charged twice"]},
    )
    assert passed is True
    assert "satisfied" in reason
    passed, reason = offline_judge(
        "status?",
        "charged twice",
        {"must_include": ["refund issued"], "must_not_include": ["charged twice"]},
    )
    assert passed is False
    assert "missing" in reason
    assert "forbidden" in reason


def test_explicit_provider_does_not_touch_the_sibling_repo(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("sibling lookup should not run")

    monkeypatch.setattr("agenteval.core.config.resolve_agent_repo", _boom)
    passed, reason = judge_correctness(
        "Refund order 4821",
        "Refund issued for order 4821",
        "Refund issued",
        provider="offline",
    )
    assert passed is True
    assert reason


def test_auto_provider_uses_offline_when_no_key_and_no_sibling(monkeypatch):
    def _missing(*args, **kwargs):
        raise AgentDependencyNotFound("not here")

    monkeypatch.setattr("agenteval.core.config.resolve_agent_repo", _missing)
    assert select_judge_provider() == "offline"
    passed, _reason = judge_correctness("q", "Refund issued", "Refund issued")
    assert passed is True


def test_auto_provider_prefers_an_api_key_over_the_sibling(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    def _boom(*args, **kwargs):
        raise AssertionError("sibling lookup should not run when a key is set")

    monkeypatch.setattr("agenteval.core.config.resolve_agent_repo", _boom)
    assert select_judge_provider() == "groq"


def test_env_provider_overrides_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AGENTEVAL_JUDGE_PROVIDER", "offline")
    assert select_judge_provider() == "offline"


def test_remote_judge_parses_a_provider_json_body(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            body = {"choices": [{"message": {"content": '{"pass": false, "reason": "no refund"}'}}]}
            return json.dumps(body).encode()

    def _open(request, timeout=0):
        assert "api.openai.com" in request.full_url
        assert request.get_header("Authorization") == "Bearer test-key"
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", _open)
    passed, reason = judge_correctness("q", "cancelled", "refund issued", provider="openai")
    assert passed is False
    assert reason == "no refund"


def test_remote_judge_reports_http_errors_without_raising(monkeypatch):
    import urllib.error

    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    def _open(request, timeout=0):
        raise urllib.error.HTTPError(
            request.full_url, 401, "unauthorized", hdrs=None, fp=io.BytesIO(b'{"error":"bad key"}')
        )

    monkeypatch.setattr("urllib.request.urlopen", _open)
    passed, reason = judge_correctness("q", "answer", "criteria", provider="groq")
    assert passed is False
    assert reason.startswith("judge error:")
    assert "401" in reason


def test_unknown_provider_is_a_judge_error_not_a_crash():
    passed, reason = judge_correctness("q", "answer", "criteria", provider="nope")
    assert passed is False
    assert "unknown provider" in reason
