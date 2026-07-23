"""Local-only HTTP API for the AgentEval dashboard backend (Tier 7, Phase 1).

Stdlib ``http.server`` only -- no new dependency. The scope is a handful of
read-only GET endpoints serving data that already exists as JSON files
(runs, trend history, calibration history); nothing here does
authentication, TLS termination, or is meant to be reachable beyond
localhost. ``core.rbac``/``core.audit`` are the separate, still
hosting-independent pieces a future real deployment wires auth around --
this module does not check permissions or write audit entries itself.

Data-fetching (``list_runs``/etc.) is plain, directly testable functions
operating on already-resolved paths; the HTTP handler is a thin routing
layer over them, kept separate so most behavior can be tested without
opening a socket at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from agenteval.core.calibration import load_calibration_history
from agenteval.core.history import DEFAULT_HISTORY_LIMIT, load_history

DEFAULT_PORT = 8765
DEFAULT_HOST = "127.0.0.1"


@dataclass(frozen=True)
class AgentPaths:
    """Already-resolved on-disk locations for one agent's dashboard data.

    Resolving these (registry lookup, relative-path-vs-registry-dir
    handling, the sidecar-root-vs-configured-runs_dir distinction) is the
    CLI layer's job -- exactly like ``agenteval report``/``compare`` already
    do via ``_configured_path``/``_history_root``/``_history_path_for`` --
    so this module never needs a registry or YAML at all.
    """

    runs_dir: Path
    history_path: Path
    calibration_dir: Path


def list_runs(runs_dir: str | Path) -> list[dict[str, Any]]:
    """Summarize every primary run JSON directly under ``runs_dir``.

    Corrupted or non-object files are skipped rather than failing the whole
    listing -- the same tolerance ``core.history``/``core.compare`` already
    apply to hand-edited or partially-written run artifacts.
    """
    directory = Path(runs_dir)
    if not directory.is_dir():
        return []
    summaries: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        summaries.append(
            {
                "file": path.name,
                "run_id": data.get("run_id"),
                "timestamp": data.get("timestamp"),
                "git_sha": data.get("git_sha"),
                "adapter": data.get("adapter"),
                "correctness_rate": data.get("correctness_rate"),
                "hallucination_rate": data.get("hallucination_rate"),
                "tool_call_accuracy": data.get("tool_call_accuracy"),
                "total_cost_usd": data.get("total_cost_usd"),
                "case_count": len(data.get("case_results") or []),
            }
        )
    return summaries


def get_trend(history_path: str | Path) -> list[dict[str, Any]]:
    """Trend-history entries, oldest first (see ``core.history.load_history``)."""
    return [entry.to_dict() for entry in load_history(history_path)]


def get_calibration_history(calibration_dir: str | Path) -> list[dict[str, Any]]:
    """Persisted calibration results, oldest first (see ``core.calibration``)."""
    return load_calibration_history(calibration_dir)


class _APIError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def make_handler_class(agent_paths: dict[str, AgentPaths]) -> type[BaseHTTPRequestHandler]:
    """Build a ``BaseHTTPRequestHandler`` bound to ``agent_paths`` via closure.

    ``http.server`` constructs one handler instance per request, so
    per-server configuration can't live on ``self`` from an ``__init__``
    override without re-plumbing the whole ``HTTPServer`` constructor chain
    -- a factory closure binding ``agent_paths`` as a class attribute is the
    standard, simplest way to inject it.
    """

    class Handler(BaseHTTPRequestHandler):
        _agent_paths = agent_paths

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass  # silence default stderr access logging; still overridable

        def _send_json(self, status: int, payload: Any) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _require_agent(self, params: dict[str, str]) -> tuple[str, AgentPaths]:
            name = params.get("agent")
            if not name:
                raise _APIError(400, "agent query parameter is required")
            paths = self._agent_paths.get(name)
            if paths is None:
                raise _APIError(404, f"unknown agent: {name}")
            return name, paths

        def _parse_limit(self, params: dict[str, str]) -> int:
            raw = params.get("limit")
            if raw is None:
                return DEFAULT_HISTORY_LIMIT
            try:
                limit = int(raw)
            except ValueError as exc:
                raise _APIError(400, f"limit must be an integer, got {raw!r}") from exc
            if limit < 1:
                raise _APIError(400, "limit must be at least 1")
            return limit

        def do_GET(self) -> None:  # noqa: N802 -- BaseHTTPRequestHandler's naming
            parsed = urlparse(self.path)
            params = {key: values[0] for key, values in parse_qs(parsed.query).items()}
            try:
                if parsed.path == "/api/health":
                    self._send_json(200, {"status": "ok"})
                elif parsed.path == "/api/runs":
                    name, paths = self._require_agent(params)
                    self._send_json(200, {"agent": name, "runs": list_runs(paths.runs_dir)})
                elif parsed.path == "/api/trend":
                    name, paths = self._require_agent(params)
                    limit = self._parse_limit(params)
                    self._send_json(
                        200, {"agent": name, "trend": get_trend(paths.history_path)[-limit:]}
                    )
                elif parsed.path == "/api/calibration-history":
                    name, paths = self._require_agent(params)
                    self._send_json(
                        200,
                        {
                            "agent": name,
                            "calibration_history": get_calibration_history(paths.calibration_dir),
                        },
                    )
                else:
                    raise _APIError(404, f"unknown endpoint: {parsed.path}")
            except _APIError as exc:
                self._send_json(exc.status, {"error": exc.message})

    return Handler


def run_server(
    agent_paths: dict[str, AgentPaths],
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> ThreadingHTTPServer:
    """Build (but do not block on) a local HTTP server for the given agents.

    Callers are responsible for ``serve_forever()``/shutdown -- kept
    separate so tests can run the server on a background thread against an
    OS-assigned ephemeral port (``port=0``) and shut it down cleanly.
    """
    handler_class = make_handler_class(agent_paths)
    return ThreadingHTTPServer((host, port), handler_class)
