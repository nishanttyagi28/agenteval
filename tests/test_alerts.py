import json

import pytest

from agenteval.core.alerts import (
    AlertError,
    build_message,
    maybe_send_regression_alert,
    send_webhook_alert,
)
from agenteval.core.compare import ComparisonResult


def failing_result(*reasons):
    return ComparisonResult(passed=False, reasons=list(reasons))


def passing_result():
    return ComparisonResult(passed=True)


# ── build_message ────────────────────────────────────────────────────────────


def test_build_message_includes_agent_name_and_reasons():
    message = build_message("my_agent", failing_result("correctness dropped 10.0pp"))

    assert "my_agent" in message
    assert "correctness dropped 10.0pp" in message


def test_build_message_appends_run_url_when_given():
    message = build_message("my_agent", failing_result("x"), run_url="https://ci.example/run/1")

    assert message.endswith("https://ci.example/run/1")


# ── send_webhook_alert ───────────────────────────────────────────────────────


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def close(self):
        pass


def test_send_webhook_alert_posts_slack_shaped_json(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["headers"] = dict(request.header_items())
        return _FakeResponse()

    monkeypatch.setattr("agenteval.core.alerts.urllib.request.urlopen", fake_urlopen)

    send_webhook_alert("https://hooks.example.test/webhook", "hello", kind="slack")

    assert captured["url"] == "https://hooks.example.test/webhook"
    assert captured["body"] == {"text": "hello"}


def test_send_webhook_alert_posts_discord_shaped_json(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse()

    monkeypatch.setattr("agenteval.core.alerts.urllib.request.urlopen", fake_urlopen)

    send_webhook_alert("https://discord.example.test/webhook", "hello", kind="discord")

    assert captured["body"] == {"content": "hello"}


def test_send_webhook_alert_wraps_network_failure_in_alert_error(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr("agenteval.core.alerts.urllib.request.urlopen", fake_urlopen)

    with pytest.raises(AlertError, match="failed to send slack alert"):
        send_webhook_alert("https://hooks.example.test/webhook", "hello")


# ── maybe_send_regression_alert ──────────────────────────────────────────────


def test_maybe_send_returns_none_when_gate_passed():
    status = maybe_send_regression_alert(
        agent_name="a",
        result=passing_result(),
        enabled=True,
        webhook_url_env="SOME_ENV",
    )
    assert status is None


def test_maybe_send_stays_silent_when_disabled():
    status = maybe_send_regression_alert(
        agent_name="a",
        result=failing_result("x"),
        enabled=False,
        webhook_url_env="SOME_ENV",
    )
    assert status is None


def test_maybe_send_skips_when_no_webhook_env_configured():
    status = maybe_send_regression_alert(
        agent_name="a",
        result=failing_result("x"),
        enabled=True,
        webhook_url_env=None,
    )
    assert status == "skipped: no webhook_url_env configured"


def test_maybe_send_skips_when_env_var_unset(monkeypatch):
    monkeypatch.delenv("AGENTEVAL_TEST_UNSET_WEBHOOK", raising=False)

    status = maybe_send_regression_alert(
        agent_name="a",
        result=failing_result("x"),
        enabled=True,
        webhook_url_env="AGENTEVAL_TEST_UNSET_WEBHOOK",
    )
    assert status == "skipped: AGENTEVAL_TEST_UNSET_WEBHOOK not set"


def test_maybe_send_reports_sent_on_success(monkeypatch):
    monkeypatch.setenv("AGENTEVAL_TEST_WEBHOOK", "https://hooks.example.test/webhook")
    monkeypatch.setattr("agenteval.core.alerts.send_webhook_alert", lambda *a, **k: None)

    status = maybe_send_regression_alert(
        agent_name="a",
        result=failing_result("x"),
        enabled=True,
        webhook_url_env="AGENTEVAL_TEST_WEBHOOK",
    )
    assert status == "sent"


def test_maybe_send_reports_error_without_raising(monkeypatch):
    monkeypatch.setenv("AGENTEVAL_TEST_WEBHOOK", "https://hooks.example.test/webhook")

    def boom(*a, **k):
        raise AlertError("failed to send slack alert: boom")

    monkeypatch.setattr("agenteval.core.alerts.send_webhook_alert", boom)

    status = maybe_send_regression_alert(
        agent_name="a",
        result=failing_result("x"),
        enabled=True,
        webhook_url_env="AGENTEVAL_TEST_WEBHOOK",
    )
    assert status == "error: failed to send slack alert: boom"
