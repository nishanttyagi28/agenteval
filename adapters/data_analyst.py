"""Backward-compatible import path for the Agentic Data Analyst adapter."""

from agenteval.adapters.agentic_data_analyst import (
    AgenticDataAnalystAdapter,
    DataAnalystAdapter,
    _usage_capture,
)

__all__ = ["AgenticDataAnalystAdapter", "DataAnalystAdapter"]
