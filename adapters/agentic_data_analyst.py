"""Concrete adapter for the custom Agentic Data Analyst orchestrator."""

from __future__ import annotations

import sys
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agenteval.adapters.base import AgentAdapter, AgentResponse
from agenteval.core.config import resolve_agent_repo

ROUTE_TO_TOOL: dict[str, str] = {
    "sql": "sql_agent",
    "ml": "ml_agent",
    "stats": "stats_agent",
    "forecast": "forecast_agent",
    "quality": "quality_agent",
    "insight": "insight_agent",
    "report": "report_agent",
    "rag": "rag_agent",
    "general": "general",
}


@dataclass
class _EmptyUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    model: str = "unknown"


def _usage_capture():
    """Use provider tracking when the agent version supports it, else remain compatible."""
    from agents import llm_client

    capture = getattr(llm_client, "capture_llm_usage", None)
    return capture() if callable(capture) else nullcontext(_EmptyUsage())


def _extract_final_answer(result: dict[str, Any]) -> str:
    for key in ("answer", "summary", "markdown", "explanation", "summary_for_rag"):
        value = result.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    error = result.get("error")
    if error is not None and str(error).strip():
        return str(error).strip()
    return str(result) if result else ""


def _tools_and_nodes(result: dict[str, Any]) -> tuple[list[str], list[str]]:
    nodes: list[str] = []
    tools: list[str] = []
    route = str(result.get("route") or "").strip().lower()
    agent = str(result.get("agent") or "").strip().lower()
    if route and route != "none":
        nodes.append(f"route:{route}")
    if agent:
        nodes.append(f"agent:{agent}")
    key = route or agent
    if key in ROUTE_TO_TOOL:
        tools.append(ROUTE_TO_TOOL[key])
    elif key:
        tools.append(key if key.endswith("_agent") else f"{key}_agent")
    return list(dict.fromkeys(tools)), list(dict.fromkeys(nodes))


class AgenticDataAnalystAdapter(AgentAdapter):
    """Load a CSV and normalize ``Orchestrator.handle_query`` output."""

    def __init__(
        self,
        csv_path: str | Path | None = None,
        *,
        business_context: str = "",
        session_id: str | None = None,
        db_path: str | Path | None = None,
        agent_repo_path: str | Path | None = None,
        repo_path: str | Path | None = None,
    ) -> None:
        if agent_repo_path is not None and repo_path is not None:
            if Path(agent_repo_path).resolve() != Path(repo_path).resolve():
                raise ValueError("agent_repo_path and repo_path disagree")
        self.agent_repo = resolve_agent_repo(agent_repo_path or repo_path)
        if str(self.agent_repo) not in sys.path:
            sys.path.insert(0, str(self.agent_repo))

        from utils.env import load_project_env

        load_project_env()
        from agents.ingestion import ingest_csv
        from agents.orchestrator import Orchestrator
        from db.database import TABLE_NAME, get_engine

        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.business_context = business_context or ""
        database = Path(db_path) if db_path else self.agent_repo / "data" / "agenteval.db"
        database.parent.mkdir(parents=True, exist_ok=True)
        self.engine = get_engine(str(database))
        self.dataframe = None
        self.tables: dict[str, Any] = {}

        if csv_path is not None:
            path = Path(csv_path)
            if not path.is_file():
                raise FileNotFoundError(f"CSV not found: {path}")
            result = ingest_csv(file_path=str(path), engine=self.engine)
            if not result.get("success"):
                raise RuntimeError(result.get("error") or f"Failed to ingest {path}")
            self.dataframe = result["dataframe"]
            self.tables = {TABLE_NAME: self.dataframe}

        self.orchestrator = Orchestrator(
            self.engine,
            self.session_id,
            dataframe=self.dataframe,
            tables=self.tables or None,
            business_context=self.business_context,
        )

    def run(self, prompt: str, **kwargs: Any) -> AgentResponse:
        started = time.perf_counter()
        with _usage_capture() as usage:
            result = self.orchestrator.handle_query(prompt)
        latency_ms = (time.perf_counter() - started) * 1000.0
        if not isinstance(result, dict):
            result = {"success": False, "error": str(result), "route": "none"}
        tools, nodes = _tools_and_nodes(result)
        raw = dict(result)
        raw["_llm_usage"] = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.prompt_tokens + usage.completion_tokens,
            "calls": usage.calls,
            "model": usage.model,
            "provider_reported": usage.calls > 0,
        }
        return AgentResponse(
            output=_extract_final_answer(result),
            tool_calls=tools,
            nodes_fired=nodes,
            prompt_tokens=usage.prompt_tokens if usage.calls > 0 else None,
            completion_tokens=usage.completion_tokens if usage.calls > 0 else None,
            latency_ms=latency_ms,
            raw=raw,
        )


# Kept here as well for callers that imported both names from the canonical module.
DataAnalystAdapter = AgenticDataAnalystAdapter
