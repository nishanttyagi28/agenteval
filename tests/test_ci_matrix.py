from agenteval.core.ci_matrix import generate_ci_matrix
from agenteval.core.registry import DEFAULT_REGISTRY_PATH


def test_ci_matrix_contains_only_enabled_agents():
    matrix = generate_ci_matrix(DEFAULT_REGISTRY_PATH)
    assert [entry["agent"] for entry in matrix["include"]] == [
        "agentic_data_analyst"
    ]
    entry = matrix["include"][0]
    assert entry["repository"] == "nishanttyagi28/agentic-data-analyst"
    assert entry["checkout_path"] == "_deps/agentic-data-analyst"
    assert "total_customers" in entry["smoke_case_ids"]
    assert "scheme_saathi" not in str(matrix)
    assert "contract_shield" not in str(matrix)
