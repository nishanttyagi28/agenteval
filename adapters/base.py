"""AgentAdapter contract — one clean interface, one concrete implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from agenteval.core.trace import TraceStep, normalize_trace_steps


def _normalize_context_chunk(item: Any) -> dict[str, Any]:
    """Accept a plain string or a mapping for one retrieved-context chunk.

    A bare string is common for adapters wrapping a simple retriever
    (``["chunk one text", "chunk two text"]``); it becomes ``{"text": ...}``
    with no ``id``. A mapping (``{"id": ..., "text": ...}``) passes through
    unchanged so retrieval-precision/recall and citation checks (which key
    off ``id``) keep working.
    """
    if isinstance(item, str):
        return {"text": item}
    if isinstance(item, dict):
        return dict(item)
    raise TypeError("retrieved_context items must be a string or a mapping")


@dataclass(init=False)
class AgentResponse:
    """Normalized, agent-agnostic result from a single invocation.

    ``final_answer`` and ``tools_called`` remain accepted constructor aliases so
    adapters written against the original single-agent API keep working.
    """

    output: str
    tool_calls: list[str] = field(default_factory=list)
    nodes_fired: list[str] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)
    retrieved_context: list[dict[str, Any]] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    trace_steps: list[TraceStep] = field(default_factory=list)

    def __init__(
        self,
        output: str | None = None,
        tool_calls: list[str] | None = None,
        nodes_fired: list[str] | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        cost_usd: float | None = None,
        latency_ms: float = 0.0,
        raw: dict[str, Any] | None = None,
        *,
        final_answer: str | None = None,
        tools_called: list[str] | None = None,
        retrieved_context: list[Any] | None = None,
        citations: list[str] | None = None,
        trace_steps: list[Any] | None = None,
    ) -> None:
        if output is not None and final_answer is not None and output != final_answer:
            raise ValueError("output and final_answer disagree")
        if tool_calls is not None and tools_called is not None and tool_calls != tools_called:
            raise ValueError("tool_calls and tools_called disagree")
        self.output = output if output is not None else (final_answer or "")
        self.tool_calls = list(tool_calls if tool_calls is not None else (tools_called or []))
        self.nodes_fired = list(nodes_fired or [])
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = (
            total_tokens
            if total_tokens is not None
            else (
                prompt_tokens + completion_tokens
                if prompt_tokens is not None and completion_tokens is not None
                else None
            )
        )
        self.cost_usd = cost_usd
        self.latency_ms = float(latency_ms)
        self.raw = dict(raw or {})
        self.retrieved_context = [_normalize_context_chunk(item) for item in (retrieved_context or [])]
        self.citations = list(citations or [])
        self.trace_steps = normalize_trace_steps(trace_steps)

    @property
    def final_answer(self) -> str:
        return self.output

    @property
    def tools_called(self) -> list[str]:
        return self.tool_calls


# Backward-compatible name used by runner.py and third-party adapters.
AgentRun = AgentResponse


class AgentAdapter(ABC):
    """Abstract interface every agent-under-test must implement.

    Implementations should measure end-to-end latency, expose observable tool
    calls and provider usage, preserve the original payload in ``raw``, and
    raise invocation/infrastructure errors rather than returning them as a
    successful answer.
    """

    @abstractmethod
    def run(self, prompt: str, **kwargs: Any) -> AgentResponse:
        """Invoke the agent and return a normalized response."""
        raise NotImplementedError
