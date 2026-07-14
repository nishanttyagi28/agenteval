"""AgentAdapter contract — one clean interface, one concrete implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentRun:
    """Normalized output from a single agent invocation."""

    final_answer: str
    tools_called: list[str] = field(default_factory=list)
    nodes_fired: list[str] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


class AgentAdapter(ABC):
    """Abstract interface every agent-under-test must implement."""

    @abstractmethod
    def run(self, prompt: str) -> AgentRun:
        """Invoke the agent; return final answer, tools/nodes, token usage, latency."""
