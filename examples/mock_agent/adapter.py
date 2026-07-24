"""Deterministic mock agent — pure Python, no LLM, no network, no API keys.

Given a fixed prompt from the bundled golden suite, returns a fixed trajectory
(tool calls + nodes + final answer) so AgentEval can score it end-to-end like a
real agent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agenteval.adapters.base import AgentAdapter, AgentResponse


# Fixed prompt → (answer, tools, nodes) table. Keys are exact golden prompts.
_TRAJECTORIES: dict[str, dict[str, Any]] = {
    "What is 2 + 2?": {
        "output": "4",
        "tool_calls": [],
        "nodes_fired": ["agent:mock"],
        "prompt_tokens": 8,
        "completion_tokens": 1,
    },
    "What is the capital of France?": {
        "output": "The capital of France is Paris.",
        "tool_calls": ["lookup_tool"],
        "nodes_fired": ["tool:lookup_tool", "agent:mock"],
        "prompt_tokens": 12,
        "completion_tokens": 8,
    },
    "How many items are in the demo inventory?": {
        "output": "There are 42 items in the demo inventory.",
        "tool_calls": ["inventory_tool"],
        "nodes_fired": ["tool:inventory_tool", "agent:mock"],
        "prompt_tokens": 14,
        "completion_tokens": 12,
    },
}


class MockAgentAdapter(AgentAdapter):
    """Network-free adapter that replays fixed trajectories for known prompts."""

    def __init__(self, repo_path: str | Path | None = None, **_: Any) -> None:
        # Mirror other demo adapters: accept (and lightly validate) repo_path.
        if repo_path is not None:
            path = Path(repo_path)
            if not path.is_dir():
                raise ValueError(f"mock agent path does not exist: {path}")

    def run(self, prompt: str, **_: Any) -> AgentResponse:
        key = (prompt or "").strip()
        scripted = _TRAJECTORIES.get(key)
        if scripted is None:
            # Unknown prompts still return a structured response so the harness
            # can score a correctness failure rather than raising.
            return AgentResponse(
                output=f"mock agent has no scripted answer for: {key!r}",
                tool_calls=[],
                nodes_fired=["agent:mock", "error:unknown_prompt"],
                prompt_tokens=0,
                completion_tokens=0,
                cost_usd=0.0,
                latency_ms=0.0,
                raw={"fixture": True, "unknown_prompt": True, "prompt": key},
            )

        return AgentResponse(
            output=str(scripted["output"]),
            tool_calls=list(scripted["tool_calls"]),
            nodes_fired=list(scripted["nodes_fired"]),
            prompt_tokens=int(scripted["prompt_tokens"]),
            completion_tokens=int(scripted["completion_tokens"]),
            cost_usd=0.0,
            latency_ms=0.0,
            raw={
                "fixture": True,
                "trajectory": list(scripted["nodes_fired"]),
                "prompt": key,
            },
        )
