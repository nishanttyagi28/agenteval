"""Optional webhook alerting on regression-gate failure (Tier 5, Phase 3).

Builds on ``core.compare``'s existing ``ComparisonResult`` -- this module
does not recompute or duplicate gate logic, CI's PR-comment posting, or
``format_markdown``. It only turns an already-decided gate failure into one
Slack- or Discord-compatible webhook POST, using stdlib ``urllib`` so no new
third-party dependency is introduced.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from agenteval.core.compare import ComparisonResult


class AlertError(Exception):
    """Raised when a webhook alert could not be sent."""


def build_message(agent_name: str, result: ComparisonResult, *, run_url: str | None = None) -> str:
    """Plain-text alert body shared by every webhook kind."""
    lines = [f"AgentEval regression gate FAILED for `{agent_name}`"]
    lines.extend(f"- {reason}" for reason in result.reasons)
    if run_url:
        lines.append(run_url)
    return "\n".join(lines)


def _payload(kind: str, message: str) -> dict[str, str]:
    if kind == "discord":
        return {"content": message}
    return {"text": message}  # slack is the default and fallback shape


def send_webhook_alert(
    webhook_url: str,
    message: str,
    *,
    kind: str = "slack",
    timeout: float = 10.0,
) -> None:
    """POST a JSON alert payload to a Slack- or Discord-compatible webhook URL.

    Raises ``AlertError`` on any network/HTTP failure; callers decide whether
    that should be fatal (it should not affect the regression gate's own
    exit code -- see ``maybe_send_regression_alert``).
    """
    data = json.dumps(_payload(kind, message)).encode("utf-8")
    request = urllib.request.Request(
        webhook_url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        urllib.request.urlopen(request, timeout=timeout).close()
    except (urllib.error.URLError, ValueError, OSError) as exc:
        raise AlertError(f"failed to send {kind} alert: {exc}") from exc


def maybe_send_regression_alert(
    *,
    agent_name: str,
    result: ComparisonResult,
    enabled: bool,
    webhook_url_env: str | None,
    kind: str = "slack",
    run_url: str | None = None,
) -> str | None:
    """Send a regression alert if configured and the gate failed.

    Returns a short status string ("sent", "skipped: ...", "error: ...") for
    the caller to print, or ``None`` when there's nothing worth reporting --
    the gate passed, or alerting was never configured for this agent (the
    common case; staying silent avoids a noisy line on every gate failure
    for agents that haven't opted in). Never raises: a broken webhook must
    not change the gate's exit code, only fail to notify about it.
    """
    if result.passed or not enabled:
        return None
    if not webhook_url_env:
        return "skipped: no webhook_url_env configured"
    webhook_url = os.environ.get(webhook_url_env)
    if not webhook_url:
        return f"skipped: {webhook_url_env} not set"
    message = build_message(agent_name, result, run_url=run_url)
    try:
        send_webhook_alert(webhook_url, message, kind=kind)
    except AlertError as exc:
        return f"error: {exc}"
    return "sent"
