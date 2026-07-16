import sys
from contextlib import contextmanager
from types import ModuleType

from agenteval.adapters.data_analyst import _usage_capture


def install_agents(monkeypatch, llm_client):
    agents = ModuleType("agents")
    agents.llm_client = llm_client
    monkeypatch.setitem(sys.modules, "agents", agents)


def test_usage_capture_falls_back_for_old_agent(monkeypatch):
    install_agents(monkeypatch, ModuleType("agents.llm_client"))
    with _usage_capture() as usage:
        assert usage.calls == 0
        assert usage.prompt_tokens == 0


def test_usage_capture_uses_new_agent_collector(monkeypatch):
    llm_client = ModuleType("agents.llm_client")

    @contextmanager
    def capture():
        yield type("Usage", (), {"calls": 2, "prompt_tokens": 10, "completion_tokens": 5})()

    llm_client.capture_llm_usage = capture
    install_agents(monkeypatch, llm_client)
    with _usage_capture() as usage:
        assert usage.calls == 2
        assert usage.prompt_tokens + usage.completion_tokens == 15
