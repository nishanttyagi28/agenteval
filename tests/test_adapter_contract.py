from dataclasses import asdict

import pytest

from agenteval.adapters.base import AgentAdapter, AgentResponse, AgentRun
from agenteval.adapters.contract_shield import ContractShieldAdapter
from agenteval.adapters.data_analyst import DataAnalystAdapter as LegacyDataAnalystAdapter
from agenteval.adapters.agentic_data_analyst import AgenticDataAnalystAdapter
from agenteval.adapters.scheme_saathi import SchemeSaathiAdapter


def test_agent_run_is_backward_compatible_alias():
    response = AgentRun(
        final_answer="done",
        tools_called=["search"],
        prompt_tokens=4,
        completion_tokens=6,
    )
    assert AgentRun is AgentResponse
    assert response.output == "done"
    assert response.tool_calls == ["search"]
    assert response.total_tokens == 10
    assert asdict(response)["output"] == "done"


def test_new_response_names_expose_legacy_properties():
    response = AgentResponse(output="ok", tool_calls=[], nodes_fired=[])
    assert response.final_answer == "ok"
    assert response.tools_called == []


def test_adapter_contract_cannot_be_instantiated_without_run():
    class IncompleteAdapter(AgentAdapter):
        pass

    with pytest.raises(TypeError):
        IncompleteAdapter()


def test_legacy_data_analyst_import_resolves_to_canonical_class():
    assert LegacyDataAnalystAdapter is AgenticDataAnalystAdapter


@pytest.mark.parametrize("adapter_class", [SchemeSaathiAdapter, ContractShieldAdapter])
def test_unconfirmed_adapter_stubs_fail_explicitly(adapter_class):
    with pytest.raises(NotImplementedError, match="has not been confirmed"):
        adapter_class().run("hello")
