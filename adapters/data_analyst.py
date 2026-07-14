"""Concrete AgentAdapter for the existing Agentic Data Analyst orchestrator.

Wraps ``Orchestrator.handle_query`` only — does not rewrite agent logic.
Routes are mapped to tool names used in golden YAML (e.g. sql → sql_agent).
Token counts are unavailable from the current LLM client and remain None.
"""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path
from typing import Any

from agenteval.adapters.base import AgentAdapter, AgentRun

# Repo root = parent of the agenteval/ package (this monorepo)
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Route / agent keys from Orchestrator → names expected by must_call_tools
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
    """Prefer human-facing fields; fall back to error or stringified raw."""
    for key in ("answer", "summary", "markdown", "explanation", "summary_for_rag"):
        val = result.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    err = result.get("error")
    if err is not None and str(err).strip():
        return str(err).strip()
    return str(result) if result else ""


def _tools_and_nodes(result: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Map orchestrator route/agent into tool names + observability nodes."""
    nodes: list[str] = []
    tools: list[str] = []

    route = (result.get("route") or "").strip().lower()
    agent = (result.get("agent") or "").strip().lower()

    if route and route not in ("none",):
        nodes.append(f"route:{route}")
    if agent:
        nodes.append(f"agent:{agent}")

    key = route or agent
    if key in ROUTE_TO_TOOL:
        tools.append(ROUTE_TO_TOOL[key])
    elif key:
        # Unknown route — surface as-is so metrics can still compare
        tools.append(key if key.endswith("_agent") else f"{key}_agent")

    # De-dupe while preserving order
    def _unique(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in items:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return _unique(tools), _unique(nodes)


class DataAnalystAdapter(AgentAdapter):
    """Load a CSV into the existing stack and run natural-language prompts."""

    def __init__(
        self,
        csv_path: str | Path | None = None,
        *,
        business_context: str = "",
        session_id: str | None = None,
        db_path: str | Path | None = None,
    ) -> None:
        """
        Parameters
        ----------
        csv_path
            Optional path to a CSV to ingest. If omitted, ``run`` still works
            but data-dependent routes will error (same as the app without upload).
        business_context
            Optional domain hint passed to the orchestrator.
        session_id
            RAG / session id; random short id if not provided.
        db_path
            SQLite path for eval isolation (default: data/agenteval.db).
        """
        from utils.env import load_project_env

        load_project_env()

        from agents.ingestion import ingest_csv
        from agents.orchestrator import Orchestrator
        from db.database import TABLE_NAME, get_engine

        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.business_context = business_context or ""

        if db_path is None:
            db_path = _REPO_ROOT / "data" / "agenteval.db"
        self.engine = get_engine(str(db_path))

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
        """Invoke ``Orchestrator.handle_query`` and normalize to AgentRun."""
        t0 = time.perf_counter()
        result = self.orchestrator.handle_query(prompt)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if not isinstance(result, dict):
            result = {"success": False, "error": str(result), "route": "none"}

        tools, nodes = _tools_and_nodes(result)
        answer = _extract_final_answer(result)

        return AgentRun(
            final_answer=answer,
            tools_called=tools,
            nodes_fired=nodes,
            # chat_completion does not surface usage; leave None (metrics may estimate later)
            prompt_tokens=None,
            completion_tokens=None,
            latency_ms=latency_ms,
            raw=result,
        )
