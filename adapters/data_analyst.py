"""Concrete adapter for the custom Agentic Data Analyst orchestrator."""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path
from typing import Any

from agenteval.adapters.base import AgentAdapter, AgentRun
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


class DataAnalystAdapter(AgentAdapter):
    """Load a CSV and normalize ``Orchestrator.handle_query`` output."""

    def __init__(
        self,
        csv_path: str | Path | None = None,
        *,
        business_context: str = "",
        session_id: str | None = None,
        db_path: str | Path | None = None,
        agent_repo_path: str | Path | None = None,
    ) -> None:
        self.agent_repo = resolve_agent_repo(agent_repo_path)
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

    def run(self, prompt: str) -> AgentRun:
        started = time.perf_counter()
        result = self.orchestrator.handle_query(prompt)
        latency_ms = (time.perf_counter() - started) * 1000.0
        if not isinstance(result, dict):
            result = {"success": False, "error": str(result), "route": "none"}
        tools, nodes = _tools_and_nodes(result)
        return AgentRun(
            final_answer=_extract_final_answer(result),
            tools_called=tools,
            nodes_fired=nodes,
            prompt_tokens=None,
            completion_tokens=None,
            latency_ms=latency_ms,
            raw=result,
        )
