from pathlib import Path

import pytest
import yaml

from agenteval.cli import _cmd_init, build_parser
from agenteval.core.init import (
    InitError,
    detect_framework,
    generate_agents_yaml,
    generate_github_workflow,
    generate_sample_golden_suite,
    next_steps_message,
    run_first_evaluation,
    scaffold_project,
)
from agenteval.core.registry import load_agent_registry
from agenteval.core.schema import load_test_cases


# --- detect_framework -------------------------------------------------------


def test_detect_framework_returns_none_for_missing_directory(tmp_path):
    assert detect_framework(tmp_path / "does-not-exist") is None


def test_detect_framework_returns_none_for_empty_directory(tmp_path):
    assert detect_framework(tmp_path) is None


def test_detect_framework_finds_crewai_in_requirements(tmp_path):
    (tmp_path / "requirements.txt").write_text("crewai>=1,<2\n", encoding="utf-8")
    assert detect_framework(tmp_path) == "crewai"


def test_detect_framework_finds_langgraph_in_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\ndependencies = ["langgraph>=0.2"]\n', encoding="utf-8"
    )
    assert detect_framework(tmp_path) == "langgraph"


def test_detect_framework_finds_autogen_via_import_scan(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "agent.py").write_text(
        "from autogen_agentchat.agents import AssistantAgent\n", encoding="utf-8"
    )
    assert detect_framework(tmp_path) == "autogen"


def test_detect_framework_finds_openai_agents_via_import_scan(tmp_path):
    (tmp_path / "main.py").write_text("from agents import Agent, Runner\n", encoding="utf-8")
    assert detect_framework(tmp_path) == "openai_agents"


def test_detect_framework_ignores_venv_and_site_packages(tmp_path):
    venv = tmp_path / ".venv" / "site-packages"
    venv.mkdir(parents=True)
    (venv / "crewai_stub.py").write_text("import crewai\n", encoding="utf-8")
    assert detect_framework(tmp_path) is None


def test_detect_framework_scan_cap_bounds_total_files_visited_not_just_read(tmp_path, monkeypatch):
    # A huge skipped tree (node_modules) must not let the scan bypass its
    # cap: the cap must count every path rglob yields, not only the ones
    # whose content actually gets read, otherwise a large skipped directory
    # makes the scan effectively unbounded (regression test for that bug).
    from agenteval.core import init as init_module

    skipped = tmp_path / "node_modules"
    skipped.mkdir()
    for i in range(init_module._MAX_SCAN_FILES + 50):
        (skipped / f"skip_{i}.py").write_text("", encoding="utf-8")
    # Sorts after every skipped file, so it's only reached once the cap has
    # already been exhausted by the skipped entries in front of it.
    (tmp_path / "zzz_real.py").write_text("import crewai\n", encoding="utf-8")

    original_rglob = Path.rglob
    visited: list = []

    def counting_rglob(self, pattern):
        for item in original_rglob(self, pattern):
            visited.append(item)
            yield item

    monkeypatch.setattr(Path, "rglob", counting_rglob)
    detect_framework(tmp_path)

    # This is the actual contract under test: the walk stops within the cap
    # regardless of how many entries are skipped along the way. (Traversal
    # order across directories isn't guaranteed, so whether the top-level
    # real signal file happens to be seen before the cap trips isn't asserted
    # here -- see test_detect_framework_ignores_venv_and_site_packages for
    # skip-vs-detect correctness.)
    assert len(visited) <= init_module._MAX_SCAN_FILES


def test_detect_framework_unreadable_manifest_does_not_raise(tmp_path, monkeypatch):
    (tmp_path / "requirements.txt").write_text("crewai\n", encoding="utf-8")

    from agenteval.core import init as init_module

    def boom(path, *args, **kwargs):
        raise OSError("simulated read failure")

    monkeypatch.setattr(init_module.Path, "read_text", boom)
    assert detect_framework(tmp_path) is None


# --- generate_agents_yaml ----------------------------------------------------


@pytest.mark.parametrize(
    "framework,adapter_path",
    [
        ("crewai", "agenteval.adapters.crewai:CrewAIAdapter"),
        ("autogen", "agenteval.adapters.autogen:AutoGenAdapter"),
        ("openai_agents", "agenteval.adapters.openai_agents:OpenAIAgentsAdapter"),
    ],
)
def test_generate_agents_yaml_known_framework_round_trips(tmp_path, framework, adapter_path):
    path = generate_agents_yaml(tmp_path, "my_agent", framework)
    assert path == tmp_path / "agents.yaml"
    text = path.read_text(encoding="utf-8")
    assert adapter_path in text
    assert "enabled: true" in text

    registry = load_agent_registry(path)
    config = registry["my_agent"]
    assert config.adapter == adapter_path
    assert config.enabled is True


def test_generate_agents_yaml_langgraph_placeholder_content(tmp_path):
    # LangGraph's first-party adapter lands in Phase 2; here we only assert the
    # generated *content* is correct. Round-trip through the real registry is
    # covered once the adapter module exists (tests/test_langgraph_adapter.py).
    path = generate_agents_yaml(tmp_path, "my_agent", "langgraph")
    text = path.read_text(encoding="utf-8")
    assert "agenteval.adapters.langgraph:LangGraphAdapter" in text
    assert "graph_import" in text


def test_generate_agents_yaml_unsupported_framework_is_disabled_but_valid(tmp_path):
    path = generate_agents_yaml(tmp_path, "my_agent", None)
    text = path.read_text(encoding="utf-8")
    assert "enabled: false" in text
    assert "agenteval.adapters.base:AgentAdapter" in text

    registry = load_agent_registry(path)
    config = registry["my_agent"]
    assert config.enabled is False
    assert config.adapter == "agenteval.adapters.base:AgentAdapter"


def test_generate_agents_yaml_rejects_invalid_agent_name(tmp_path):
    with pytest.raises(InitError, match="Invalid agent name"):
        generate_agents_yaml(tmp_path, "Not Valid", "crewai")


def test_generate_agents_yaml_refuses_to_overwrite_without_force(tmp_path):
    generate_agents_yaml(tmp_path, "my_agent", "crewai")
    with pytest.raises(InitError, match="already exists"):
        generate_agents_yaml(tmp_path, "my_agent", "autogen")


def test_generate_agents_yaml_force_overwrites(tmp_path):
    generate_agents_yaml(tmp_path, "my_agent", "crewai")
    path = generate_agents_yaml(tmp_path, "my_agent", "autogen", force=True)
    text = path.read_text(encoding="utf-8")
    assert "agenteval.adapters.autogen:AutoGenAdapter" in text


def test_generate_agents_yaml_target_path_is_a_directory_raises(tmp_path):
    (tmp_path / "agents.yaml").mkdir()
    # A directory in the way is a collision like any other file — refused
    # without --force, exactly like the file-exists case.
    with pytest.raises(InitError, match="already exists"):
        generate_agents_yaml(tmp_path, "my_agent", "crewai")
    # Forcing past it still can't turn a directory into a file.
    with pytest.raises(OSError):
        generate_agents_yaml(tmp_path, "my_agent", "crewai", force=True)


# --- generate_sample_golden_suite -------------------------------------------


def test_generate_sample_golden_suite_is_loadable(tmp_path):
    path = generate_sample_golden_suite(tmp_path, "my_agent")
    assert path == tmp_path / "tests" / "golden" / "my_agent.yaml"
    cases = load_test_cases(path)
    assert len(cases) == 3
    assert {case.id for case in cases} == {
        "sample_exact_answer",
        "sample_numeric_answer",
        "sample_contains_answer",
    }


def test_generate_sample_golden_suite_refuses_to_overwrite_without_force(tmp_path):
    generate_sample_golden_suite(tmp_path, "my_agent")
    with pytest.raises(InitError, match="already exists"):
        generate_sample_golden_suite(tmp_path, "my_agent")


def test_generate_sample_golden_suite_force_overwrites(tmp_path):
    generate_sample_golden_suite(tmp_path, "my_agent")
    path = generate_sample_golden_suite(tmp_path, "my_agent", force=True)
    assert path.is_file()


# --- generate_github_workflow -------------------------------------------------


def test_generate_github_workflow_is_valid_yaml_and_references_action(tmp_path):
    path = generate_github_workflow(tmp_path, "my_agent", "autogen")
    assert path == tmp_path / ".github" / "workflows" / "agenteval.yml"
    text = path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(text)
    assert "nishanttyagi28/agenteval@v1" in text
    assert "install-extras: autogen" in text
    assert parsed["jobs"]["evaluate"]["steps"][1]["with"]["agent"] == "my_agent"


def test_generate_github_workflow_no_framework_omits_install_extras(tmp_path):
    path = generate_github_workflow(tmp_path, "my_agent", None)
    text = path.read_text(encoding="utf-8")
    assert "install-extras" not in text
    yaml.safe_load(text)  # still must be valid YAML


def test_generate_github_workflow_refuses_to_overwrite_without_force(tmp_path):
    generate_github_workflow(tmp_path, "my_agent", "crewai")
    with pytest.raises(InitError, match="already exists"):
        generate_github_workflow(tmp_path, "my_agent", "crewai")


# --- scaffold_project ---------------------------------------------------------


def test_scaffold_project_auto_detects_and_writes_all_three_files(tmp_path):
    (tmp_path / "requirements.txt").write_text("crewai\n", encoding="utf-8")
    plan = scaffold_project(tmp_path, "my_agent")
    assert plan.framework == "crewai"
    assert plan.agents_yaml_path.is_file()
    assert plan.golden_suite_path.is_file()
    assert plan.workflow_path.is_file()


def test_scaffold_project_explicit_none_skips_detection_even_if_detectable(tmp_path):
    (tmp_path / "requirements.txt").write_text("crewai\n", encoding="utf-8")
    plan = scaffold_project(tmp_path, "my_agent", framework=None)
    assert plan.framework is None


def test_scaffold_project_rejects_unknown_framework(tmp_path):
    with pytest.raises(InitError, match="Unknown framework"):
        scaffold_project(tmp_path, "my_agent", framework="bogus")


def test_next_steps_message_mentions_agent_and_files(tmp_path):
    plan = scaffold_project(tmp_path, "my_agent", framework="crewai")
    message = next_steps_message(plan)
    assert "my_agent" in message
    assert str(plan.agents_yaml_path) in message
    assert "CrewAI" in message


def test_next_steps_message_no_framework_points_to_manual_adapter(tmp_path):
    plan = scaffold_project(tmp_path, "my_agent", framework=None)
    message = next_steps_message(plan)
    assert "No supported framework was auto-detected" in message
    assert "subclass agenteval.adapters.base.AgentAdapter" in message


# --- run_first_evaluation -----------------------------------------------------


def test_run_first_evaluation_skips_when_framework_disabled(tmp_path, capsys):
    scaffold_project(tmp_path, "my_agent", framework=None)
    result = run_first_evaluation(tmp_path, "my_agent")
    assert result is None
    assert "first-run skipped" in capsys.readouterr().out


def test_run_first_evaluation_skips_when_registry_missing(tmp_path, capsys):
    result = run_first_evaluation(tmp_path, "my_agent")
    assert result is None
    assert "first-run skipped" in capsys.readouterr().out


def test_run_first_evaluation_reports_success_via_mocked_run(tmp_path, monkeypatch):
    scaffold_project(tmp_path, "my_agent", framework="crewai")

    def fake_run(args, config, registry_path):
        return {"errors": 0}

    monkeypatch.setattr("agenteval.cli._run_registered_agent", fake_run)
    assert run_first_evaluation(tmp_path, "my_agent") == 0


def test_run_first_evaluation_quiet_flag_only_controls_run_verbosity(tmp_path, monkeypatch):
    # Regression test: `quiet` must not silently imply --no-llm-judge -- those
    # are unrelated concerns and conflating them was a real bug caught here.
    scaffold_project(tmp_path, "my_agent", framework="crewai")
    captured = {}

    def fake_run(args, config, registry_path):
        captured["quiet"] = args.quiet
        captured["no_llm_judge"] = args.no_llm_judge
        return {"errors": 0}

    monkeypatch.setattr("agenteval.cli._run_registered_agent", fake_run)

    run_first_evaluation(tmp_path, "my_agent", quiet=False)
    assert captured == {"quiet": False, "no_llm_judge": False}

    run_first_evaluation(tmp_path, "my_agent", quiet=True)
    assert captured == {"quiet": True, "no_llm_judge": False}


def test_run_first_evaluation_reports_failure_via_mocked_run(tmp_path, monkeypatch):
    scaffold_project(tmp_path, "my_agent", framework="crewai")

    def fake_run(args, config, registry_path):
        return {"errors": 2}

    monkeypatch.setattr("agenteval.cli._run_registered_agent", fake_run)
    assert run_first_evaluation(tmp_path, "my_agent") == 1


def test_run_first_evaluation_handles_dependency_not_found(tmp_path, monkeypatch):
    scaffold_project(tmp_path, "my_agent", framework="crewai")

    from agenteval.core.config import AgentDependencyNotFound

    def fake_run(args, config, registry_path):
        raise AgentDependencyNotFound("nope")

    monkeypatch.setattr("agenteval.cli._run_registered_agent", fake_run)
    result = run_first_evaluation(tmp_path, "my_agent")
    assert result is None


# --- CLI wiring ----------------------------------------------------------------


def test_cli_init_scaffolds_project(tmp_path, capsys):
    args = build_parser().parse_args(
        ["init", "--path", str(tmp_path), "--agent-name", "my_agent", "--framework", "none"]
    )
    assert _cmd_init(args) == 0
    assert (tmp_path / "agents.yaml").is_file()
    assert (tmp_path / "tests" / "golden" / "my_agent.yaml").is_file()
    assert (tmp_path / ".github" / "workflows" / "agenteval.yml").is_file()
    out = capsys.readouterr().out
    assert "agent_name=my_agent" in out
    assert "framework=none (unsupported/not detected)" in out


def test_cli_init_rejects_collision_without_force(tmp_path):
    args = build_parser().parse_args(
        ["init", "--path", str(tmp_path), "--agent-name", "my_agent", "--framework", "none"]
    )
    assert _cmd_init(args) == 0
    assert _cmd_init(args) == 2


def test_cli_init_force_allows_rerun(tmp_path):
    base_args = ["init", "--path", str(tmp_path), "--agent-name", "my_agent", "--framework", "none"]
    assert _cmd_init(build_parser().parse_args(base_args)) == 0
    assert _cmd_init(build_parser().parse_args(base_args + ["--force"])) == 0


def test_cli_init_quiet_suppresses_output(tmp_path, capsys):
    args = build_parser().parse_args(
        ["init", "--path", str(tmp_path), "--agent-name", "my_agent", "--framework", "none", "--quiet"]
    )
    assert _cmd_init(args) == 0
    assert capsys.readouterr().out == ""


def test_cli_init_run_flag_does_not_crash_without_dependency(tmp_path, capsys):
    args = build_parser().parse_args(
        [
            "init",
            "--path",
            str(tmp_path),
            "--agent-name",
            "my_agent",
            "--framework",
            "none",
            "--run",
        ]
    )
    assert _cmd_init(args) == 0
    assert "first-run skipped" in capsys.readouterr().out


def test_cli_init_creates_missing_parent_directories(tmp_path):
    target = tmp_path / "nested" / "project"
    args = build_parser().parse_args(
        ["init", "--path", str(target), "--agent-name", "my_agent", "--framework", "none"]
    )
    assert _cmd_init(args) == 0
    assert (target / "agents.yaml").is_file()


def test_cli_init_target_path_is_a_file_reports_error(tmp_path, capsys):
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    args = build_parser().parse_args(
        ["init", "--path", str(blocked), "--agent-name", "my_agent", "--framework", "none"]
    )
    assert _cmd_init(args) == 2
    assert "error:" in capsys.readouterr().err
