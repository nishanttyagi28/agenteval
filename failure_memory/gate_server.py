"""Local HTTP server for the Release Desk.

Same constraint as ``core.server``: no authentication and no TLS. Bind it to
loopback unless the operator passes ``--allow-remote`` and understands the desk
can approve CI tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from agenteval import __version__
from agenteval.failure_memory.desk import ReleaseDesk
from agenteval.failure_memory.desk_page import render_page
from agenteval.failure_memory.intake import sample_refund_incident
from agenteval.failure_memory.review import ReviewError
from agenteval.failure_memory.schema import SchemaValidationError

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8741
_MAX_BODY = 1_000_000


@dataclass(frozen=True)
class GatePaths:
    db_path: Path
    suite_path: Path


class _APIError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def make_handler_class(paths: GatePaths) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        _paths = paths

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _send(self, status: int, payload: Any, *, content_type: str = "application/json") -> None:
            if content_type.startswith("application/json"):
                body = json.dumps(payload, default=str).encode("utf-8")
            else:
                body = payload if isinstance(payload, bytes) else str(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _desk(self) -> ReleaseDesk:
            return ReleaseDesk(self._paths.db_path, suite_path=self._paths.suite_path)

        def _read_json(self) -> Any:
            raw_length = self.headers.get("Content-Length", "0")
            try:
                length = int(raw_length)
            except ValueError as exc:
                raise _APIError(400, "Content-Length must be an integer") from exc
            if length < 0 or length > _MAX_BODY:
                raise _APIError(413, "request body is too large")
            raw = self.rfile.read(length) if length else b""
            if not raw:
                return {}
            try:
                return json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise _APIError(400, "body must be JSON") from exc

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path in ("/", "/index.html"):
                    page = render_page(__version__)
                    self._send(200, page, content_type="text/html; charset=utf-8")
                    return
                if parsed.path == "/api/gate/health":
                    self._send(200, {"status": "ok", "version": __version__})
                    return
                with self._desk() as desk:
                    if parsed.path == "/api/gate/summary":
                        self._send(200, desk.summary())
                    elif parsed.path == "/api/gate/queue":
                        self._send(200, {"items": desk.queue()})
                    elif parsed.path.startswith("/api/gate/candidates/"):
                        candidate_id = unquote(parsed.path.rsplit("/", 1)[-1])
                        self._send(200, desk.detail(candidate_id))
                    else:
                        raise _APIError(404, f"unknown endpoint: {parsed.path}")
            except _APIError as exc:
                self._send(exc.status, {"error": exc.message})
            except KeyError:
                self._send(404, {"error": "unknown candidate"})
            except (ReviewError, SchemaValidationError, ValueError) as exc:
                self._send(400, {"error": str(exc)})

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/gate/sample":
                    body = sample_refund_incident()
                    with self._desk() as desk:
                        self._send(200, desk.ingest(body, actor="release-desk"))
                    return
                payload = self._read_json()
                with self._desk() as desk:
                    if parsed.path == "/api/gate/ingest":
                        self._send(200, desk.ingest(payload, actor=_actor(payload)))
                        return
                    if parsed.path.startswith("/api/gate/candidates/"):
                        candidate_id, action = _candidate_action(parsed.path)
                        self._send(200, _mutate(desk, candidate_id, action, payload))
                        return
                    raise _APIError(404, f"unknown endpoint: {parsed.path}")
            except _APIError as exc:
                self._send(exc.status, {"error": exc.message})
            except KeyError:
                self._send(404, {"error": "unknown candidate"})
            except (ReviewError, SchemaValidationError, ValueError) as exc:
                self._send(400, {"error": str(exc)})

    return Handler


def _actor(payload: Any) -> str:
    if isinstance(payload, dict):
        actor = payload.get("actor")
        if isinstance(actor, str) and actor.strip():
            return actor.strip()[:120]
    return "release-desk"


def _candidate_action(path: str) -> tuple[str, str]:
    parts = [unquote(part) for part in path.split("/") if part]
    if len(parts) != 5 or parts[:3] != ["api", "gate", "candidates"] or not parts[3]:
        raise _APIError(404, f"unknown endpoint: {path}")
    return parts[3], parts[4]


def _mutate(desk: ReleaseDesk, candidate_id: str, action: str, payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise _APIError(400, "body must be a JSON object")
    actor = _actor(payload)
    note = payload.get("note") if isinstance(payload.get("note"), str) else None
    if action == "ship":
        behaviour = payload.get("expected_behaviour")
        case_id = payload.get("stable_case_id")
        if case_id is not None and not isinstance(case_id, str):
            raise _APIError(400, "stable_case_id must be a string")
        return desk.ship(
            candidate_id,
            behaviour if isinstance(behaviour, dict) else None,
            actor=actor,
            note=note,
            stable_case_id=case_id,
        )
    if action == "reject":
        return desk.reject(candidate_id, actor=actor, note=note)
    if action == "reopen":
        return desk.reopen(candidate_id, actor=actor, note=note)
    raise _APIError(404, f"unknown action: {action}")


def run_gate_server(
    db_path: str | Path,
    suite_path: str | Path,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> ThreadingHTTPServer:
    handler = make_handler_class(GatePaths(db_path=Path(db_path), suite_path=Path(suite_path)))
    return ThreadingHTTPServer((host, port), handler)
