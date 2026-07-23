from __future__ import annotations

from pathlib import Path

import yaml

from agenteval.cli import _cmd_generate_adversarial, build_parser


def parse(*argv):
    return build_parser().parse_args(list(argv))


def _write_cases(path: Path) -> Path:
    cases = [
        {
            "id": "c1",
            "prompt": "What is the capital of France?",
            "expects": {"correctness_type": "contains", "ground_truth": "Paris"},
        },
        {
            "id": "c2",
            "prompt": "What is 2+2?",
            "expects": {"correctness_type": "numeric", "ground_truth": 4},
        },
    ]
    path.write_text(yaml.safe_dump(cases, sort_keys=False), encoding="utf-8")
    return path


def test_generate_adversarial_is_registered():
    assert parse("generate-adversarial").func is _cmd_generate_adversarial


def test_happy_path_writes_yaml_and_reports_counts(tmp_path, capsys):
    source = _write_cases(tmp_path / "cases.yaml")
    output = tmp_path / "out.yaml"
    args = parse(
        "generate-adversarial", "--from", str(source), "--output", str(output)
    )
    assert _cmd_generate_adversarial(args) == 0
    out = capsys.readouterr().out
    assert "generated=8" in out  # 2 source cases x 4 strategies
    assert "source_cases=2" in out
    assert f"candidates={output}" in out
    assert "best-effort robustness probes" in out
    assert output.is_file()


def test_strategies_flag_reduces_generated_count(tmp_path, capsys):
    source = _write_cases(tmp_path / "cases.yaml")
    output = tmp_path / "out.yaml"
    args = parse(
        "generate-adversarial",
        "--from", str(source),
        "--strategies", "ambiguous_qualifier,contradictory_context",
        "--output", str(output),
    )
    assert _cmd_generate_adversarial(args) == 0
    out = capsys.readouterr().out
    assert "generated=4" in out  # 2 source cases x 2 strategies


def test_case_id_filters_source_cases(tmp_path, capsys):
    source = _write_cases(tmp_path / "cases.yaml")
    output = tmp_path / "out.yaml"
    args = parse(
        "generate-adversarial",
        "--from", str(source),
        "--case-id", "c1",
        "--output", str(output),
    )
    assert _cmd_generate_adversarial(args) == 0
    out = capsys.readouterr().out
    assert "generated=4" in out  # 1 source case x 4 strategies
    assert "source_cases=1" in out


def test_unknown_case_id_selection_errors(tmp_path, capsys):
    source = _write_cases(tmp_path / "cases.yaml")
    output = tmp_path / "out.yaml"
    args = parse(
        "generate-adversarial",
        "--from", str(source),
        "--case-id", "does-not-exist",
        "--output", str(output),
    )
    assert _cmd_generate_adversarial(args) == 2
    assert "error:" in capsys.readouterr().err


def test_overwrite_refusal_then_force(tmp_path, capsys):
    source = _write_cases(tmp_path / "cases.yaml")
    output = tmp_path / "out.yaml"
    output.write_text("existing content", encoding="utf-8")

    args = parse("generate-adversarial", "--from", str(source), "--output", str(output))
    assert _cmd_generate_adversarial(args) == 2
    err = capsys.readouterr().err
    assert "use --overwrite" in err

    args = parse(
        "generate-adversarial", "--from", str(source), "--output", str(output), "--overwrite"
    )
    assert _cmd_generate_adversarial(args) == 0


def test_unknown_strategy_errors_with_exit_2(tmp_path, capsys):
    source = _write_cases(tmp_path / "cases.yaml")
    output = tmp_path / "out.yaml"
    args = parse(
        "generate-adversarial",
        "--from", str(source),
        "--strategies", "not_a_real_strategy",
        "--output", str(output),
    )
    assert _cmd_generate_adversarial(args) == 2
    assert "error:" in capsys.readouterr().err
