from pathlib import Path

import yaml


def test_eval_workflow_uses_generated_agent_matrix():
    workflow_path = Path(__file__).parents[1] / ".github" / "workflows" / "eval.yml"
    text = workflow_path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(text)
    jobs = parsed["jobs"]

    assert "prepare-matrix" in jobs
    assert "python -m agenteval.core.ci_matrix" in text
    assert "fromJSON(needs.prepare-matrix.outputs.matrix)" in text
    assert 'python -m agenteval run --agent "${{ matrix.agent }}"' in text
    assert "python -m agenteval compare" in text
    assert '--agent "${{ matrix.agent }}"' in text
    assert "steps.dependency.outputs.available == 'true'" in text
    assert "repeat_count:" in text
    assert "repeat_case:" in text
    assert 'repeat_args=(--repeat "$repeat_count" --repeat-case "$repeat_case")' in text
    assert 'case_ids+=("$repeat_case")' in text
    assert 'selected.add(os.environ["REPEAT_CASE"])' in text
    assert "find \"$artifact_dir/current\" -maxdepth 1" in text
