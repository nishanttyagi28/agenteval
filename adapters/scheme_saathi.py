"""Scheme Saathi adapter placeholder pending entrypoint confirmation."""

from typing import Any

from agenteval.adapters.base import AgentAdapter, AgentResponse


class SchemeSaathiAdapter(AgentAdapter):
    """TODO: implement after Scheme Saathi's real invocation interface is confirmed."""

    def run(self, prompt: str, **kwargs: Any) -> AgentResponse:
        raise NotImplementedError(
            "Scheme Saathi invocation interface has not been confirmed"
        )
