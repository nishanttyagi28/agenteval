"""Contract Shield adapter placeholder pending entrypoint confirmation."""

from typing import Any

from agenteval.adapters.base import AgentAdapter, AgentResponse


class ContractShieldAdapter(AgentAdapter):
    """TODO: implement after Contract Shield's real invocation interface is confirmed."""

    def run(self, prompt: str, **kwargs: Any) -> AgentResponse:
        raise NotImplementedError(
            "Contract Shield invocation interface has not been confirmed"
        )
