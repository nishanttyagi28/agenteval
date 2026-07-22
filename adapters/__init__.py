"""Agent adapters for AgentEval."""

from agenteval.adapters.base import AgentAdapter, AgentResponse, AgentRun
from agenteval.adapters.agentic_data_analyst import (
    AgenticDataAnalystAdapter,
    DataAnalystAdapter,
)
from agenteval.adapters.crewai import CrewAIAdapter
from agenteval.adapters.autogen import AutoGenAdapter

__all__ = [
    "AgentAdapter",
    "AgentResponse",
    "AgentRun",
    "AgenticDataAnalystAdapter",
    "DataAnalystAdapter",
    "CrewAIAdapter",
    "AutoGenAdapter",
]
