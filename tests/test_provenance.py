import hashlib

from agenteval.core.provenance import collect_provenance, sha256_file


def test_sha256_file(tmp_path):
    path = tmp_path / "fixture.txt"
    path.write_bytes(b"AgentEval")
    assert sha256_file(path) == hashlib.sha256(b"AgentEval").hexdigest()


def test_collect_provenance_contains_reproducibility_fields(tmp_path):
    cases = tmp_path / "cases.yaml"
    data = tmp_path / "data.csv"
    cases.write_text("[]", encoding="utf-8")
    data.write_text("a\n1\n", encoding="utf-8")
    result = collect_provenance(
        agenteval_repo=tmp_path,
        agent_repo=tmp_path,
        cases_path=cases,
        dataset_path=data,
    )
    assert len(result["golden_suite_sha256"]) == 64
    assert len(result["dataset_sha256"]) == 64
    assert result["python_version"]
