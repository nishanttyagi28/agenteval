from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_root_action_is_a_well_formed_composite_action():
    action = load_yaml(ROOT / "action.yml")

    assert action["runs"]["using"] == "composite"
    assert {"agent", "config-file", "agent-path", "cases-file"} <= set(action["inputs"])
    assert {"passed", "report-path", "comparison-path"} == set(action["outputs"])
    assert all("shell" in step or "uses" in step for step in action["runs"]["steps"])
    text = (ROOT / "action.yml").read_text(encoding="utf-8")
    assert "python -m agenteval run" in text
    assert "python -m agenteval compare" in text
    assert re.search(r"(?m)^\s*eval\s", text) is None


def test_action_smoke_workflow_consumes_local_composite_action():
    workflow = load_yaml(ROOT / ".github" / "workflows" / "action-smoke.yml")
    steps = workflow["jobs"]["smoke"]["steps"]

    action_step = next(step for step in steps if step.get("id") == "agenteval")
    assert action_step["uses"] == "./"
    assert action_step["with"]["agent"] == "action_demo"
    assert any(step.get("name") == "Verify action outputs" for step in steps)


def test_consumer_example_uses_versioned_root_action():
    workflow = load_yaml(ROOT / "examples" / "github-actions" / "agenteval.yml")
    steps = workflow["jobs"]["evaluate"]["steps"]

    action_step = next(step for step in steps if step.get("id") == "agenteval")
    assert action_step["uses"] == "nishanttyagi28/agenteval@v1"
    assert action_step["with"]["agent"] == "my_agent"
