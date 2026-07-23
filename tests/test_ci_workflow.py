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


def test_eval_workflow_generates_and_links_html_report():
    workflow_path = Path(__file__).parents[1] / ".github" / "workflows" / "eval.yml"
    text = workflow_path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(text)
    steps = parsed["jobs"]["live-eval"]["steps"]

    report_step = next(step for step in steps if step.get("name") == "Generate HTML report")
    assert report_step["if"] == (
        "always() && steps.preflight.outputs.has_secret == 'true' && "
        "steps.dependency.outputs.available == 'true'"
    )
    assert "python -m agenteval report" in report_step["run"]
    assert '--run "$current"' in report_step["run"]
    assert '--baseline "$artifact_dir/baseline.json"' in report_step["run"]
    assert '--output "$artifact_dir/report.html"' in report_step["run"]

    # report.html lands inside the same artifact_dir the existing upload-artifact
    # step already uploads wholesale, so no separate upload step is needed.
    upload_step = next(step for step in steps if step.get("name") == "Upload evaluation evidence")
    assert upload_step["with"]["path"] == "artifacts/${{ matrix.agent }}/"

    comment_step = next(step for step in steps if step.get("name") == "Update pull request report")
    script = comment_step["with"]["script"]
    assert "Full HTML report" in script
    assert "actions/runs/${{ github.run_id }}" in script
    assert "agenteval-${{ matrix.agent }}-${{ github.sha }}" in script
