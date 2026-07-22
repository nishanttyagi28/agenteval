"""Network-free demo adapter for validating the reusable GitHub Action."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agenteval.adapters.base import AgentAdapter, AgentResponse


class ActionDemoAdapter(AgentAdapter):
    def __init__(self, repo_path: str | Path, **_: Any) -> None:
        path = Path(repo_path)
        if not path.is_dir():
            raise ValueError(f"demo agent path does not exist: {path}")

    def run(self, prompt: str, **_: Any) -> AgentResponse:
        return AgentResponse(
            output=f"AgentEval action received: {prompt}",
            tool_calls=["demo_tool"],
            nodes_fired=["agent:action_demo"],
            prompt_tokens=5,
            completion_tokens=7,
            cost_usd=0.0,
            raw={"fixture": True},
        )

