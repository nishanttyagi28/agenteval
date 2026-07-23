from pathlib import Path

from agenteval.cli import _cmd_calibrate, build_parser
from agenteval.core.schema import AgentConfig, GateConfig, RepositoryConfig


def make_config(tmp_path: Path, name: str = "example_agent") -> AgentConfig:
    return AgentConfig(
        name=name,
        display_name="Example Agent",
        adapter="agenteval.adapters.scheme_saathi:SchemeSaathiAdapter",
        repository=RepositoryConfig(env_var=f"{name.upper()}_PATH"),
        golden_suite=tmp_path / "golden.yaml",
        baseline=tmp_path / "baseline.json",
        runs_dir=tmp_path / "runs",
        gates=GateConfig(),
    )


def setup_registry(tmp_path, monkeypatch, config=None):
    config = config or make_config(tmp_path)
    monkeypatch.setattr(
        "agenteval.core.registry.load_agent_registry", lambda path: {config.name: config}
    )
    monkeypatch.setattr(
        "agenteval.core.registry.resolve_agent_repository",
        lambda config, **kwargs: tmp_path / "sibling_repo",
    )
    return config


def write_golden_set(tmp_path, *, labels):
    lines = []
    for index, label in enumerate(labels):
        lines.append(
            f"- id: c{index}\n"
            f"  prompt: \"prompt {index}\"\n"
            f"  ground_truth: \"gt {index}\"\n"
            f"  candidate_answer: \"answer {index}\"\n"
            f"  human_label: {'true' if label else 'false'}\n"
        )
    path = tmp_path / "calibration.yaml"
    path.write_text("".join(lines), encoding="utf-8")
    return path


def parse(argv):
    return build_parser().parse_args(["calibrate", *argv])


def test_calibrate_perfect_agreement_exits_zero(tmp_path, monkeypatch, capsys):
    setup_registry(tmp_path, monkeypatch)
    golden_set = write_golden_set(tmp_path, labels=[True, False, True, False])
    monkeypatch.setattr(
        "agenteval.core.judge.judge_correctness",
        lambda prompt, answer, gt, **kwargs: (answer.endswith(("0", "2")), "matched"),
    )

    args = parse(
        ["--judge", "example_agent", "--golden-set", str(golden_set), "--registry", str(tmp_path / "agents.yaml")]
    )
    assert _cmd_calibrate(args) == 0

    out = capsys.readouterr().out
    assert "judge=example_agent" in out
    assert "n_cases=4" in out
    assert "cohens_kappa=1.000" in out


def test_calibrate_reports_mismatches_and_nonzero_exit_below_threshold(tmp_path, monkeypatch, capsys):
    setup_registry(tmp_path, monkeypatch)
    golden_set = write_golden_set(tmp_path, labels=[True, True, False, False])
    # Judge always says "pass" regardless of the real label -> 2/4 agreement.
    monkeypatch.setattr(
        "agenteval.core.judge.judge_correctness",
        lambda prompt, answer, gt, **kwargs: (True, "always pass"),
    )

    args = parse(
        ["--judge", "example_agent", "--golden-set", str(golden_set), "--registry", str(tmp_path / "agents.yaml")]
    )
    exit_code = _cmd_calibrate(args)

    out = capsys.readouterr().out
    assert "mismatches=c2,c3" in out
    assert exit_code == 1


def test_calibrate_unknown_agent_is_a_clean_error(tmp_path, monkeypatch, capsys):
    setup_registry(tmp_path, monkeypatch)
    golden_set = write_golden_set(tmp_path, labels=[True])

    args = parse(
        ["--judge", "does_not_exist", "--golden-set", str(golden_set), "--registry", str(tmp_path / "agents.yaml")]
    )
    assert _cmd_calibrate(args) == 2
    assert "Unknown agent" in capsys.readouterr().err


def test_calibrate_missing_golden_set_is_a_clean_error(tmp_path, monkeypatch, capsys):
    setup_registry(tmp_path, monkeypatch)

    args = parse(
        [
            "--judge",
            "example_agent",
            "--golden-set",
            str(tmp_path / "nope.yaml"),
            "--registry",
            str(tmp_path / "agents.yaml"),
        ]
    )
    assert _cmd_calibrate(args) == 2
    assert "error:" in capsys.readouterr().err


def test_calibrate_empty_golden_set_is_a_clean_error(tmp_path, monkeypatch, capsys):
    setup_registry(tmp_path, monkeypatch)
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")

    args = parse(
        ["--judge", "example_agent", "--golden-set", str(empty), "--registry", str(tmp_path / "agents.yaml")]
    )
    assert _cmd_calibrate(args) == 2
    assert "calibration set is empty" in capsys.readouterr().err


def test_calibrate_custom_kappa_threshold_flag(tmp_path, monkeypatch, capsys):
    setup_registry(tmp_path, monkeypatch)
    golden_set = write_golden_set(tmp_path, labels=[True, False])
    monkeypatch.setattr(
        "agenteval.core.judge.judge_correctness",
        lambda prompt, answer, gt, **kwargs: (True, "always pass"),
    )

    args = parse(
        [
            "--judge",
            "example_agent",
            "--golden-set",
            str(golden_set),
            "--registry",
            str(tmp_path / "agents.yaml"),
            "--kappa-threshold",
            "-1.0",
        ]
    )
    assert _cmd_calibrate(args) == 0  # -1.0 threshold: nothing can be "below" it
