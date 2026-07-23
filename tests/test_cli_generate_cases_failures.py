import json
from pathlib import Path

from agenteval.cli import _cmd_generate_cases, build_parser


def run_report(*case_results):
    return {"run_id": "r1", "timestamp": "2026-01-01T00:00:00Z", "case_results": list(case_results)}


def case(case_id, prompt, final_answer, *, status="passed", **overrides):
    return {
        "case_id": case_id,
        "prompt": prompt,
        "final_answer": final_answer,
        "status": status,
        **overrides,
    }


def write_report(path: Path, report: dict) -> Path:
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def parse(argv):
    return build_parser().parse_args(["generate-cases", *argv])


def test_from_failures_writes_candidate_yaml(tmp_path):
    baseline_path = write_report(
        tmp_path / "baseline.json", run_report(case("c1", "How many?", "42", status="passed"))
    )
    current_path = write_report(
        tmp_path / "current.json",
        run_report(case("c1", "How many?", "", status="failed", raw={"error": "wrong"})),
    )
    output = tmp_path / "candidates.yaml"

    args = parse(
        [
            "--from-failures",
            "--baseline",
            str(baseline_path),
            "--current",
            str(current_path),
            "--output",
            str(output),
        ]
    )
    assert _cmd_generate_cases(args) == 0
    assert output.is_file()
    assert "c1__regression" in output.read_text(encoding="utf-8")


def test_from_failures_requires_baseline_and_current(tmp_path, capsys):
    args = parse(["--from-failures", "--output", str(tmp_path / "out.yaml")])
    assert _cmd_generate_cases(args) == 2
    assert "requires both --baseline and --current" in capsys.readouterr().err


def test_from_failures_cannot_combine_with_logs(tmp_path, capsys):
    baseline_path = write_report(tmp_path / "b.json", run_report())
    current_path = write_report(tmp_path / "c.json", run_report())
    args = parse(
        [
            "--from-failures",
            "--baseline",
            str(baseline_path),
            "--current",
            str(current_path),
            "--logs",
            str(tmp_path / "logs.json"),
            "--output",
            str(tmp_path / "out.yaml"),
        ]
    )
    assert _cmd_generate_cases(args) == 2
    assert "cannot be combined with --logs" in capsys.readouterr().err


def test_logs_mode_still_works_without_from_failures(tmp_path):
    logs_path = write_report(
        tmp_path / "logs.json", run_report(case("c1", "How many?", "42", status="passed"))
    )
    output = tmp_path / "candidates.yaml"

    args = parse(["--logs", str(logs_path), "--output", str(output)])
    assert _cmd_generate_cases(args) == 0
    assert output.is_file()


def test_no_mode_selected_is_a_clean_error(tmp_path, capsys):
    args = parse(["--output", str(tmp_path / "out.yaml")])
    assert _cmd_generate_cases(args) == 2
    assert "--logs is required unless --from-failures is used" in capsys.readouterr().err


def test_from_failures_similarity_threshold_flag_is_passed_through(tmp_path):
    baseline_path = write_report(
        tmp_path / "baseline.json",
        run_report(
            case("a", "q1", "answer 1", status="passed"),
            case("b", "q2", "answer 2", status="passed"),
        ),
    )
    current_path = write_report(
        tmp_path / "current.json",
        run_report(
            case("a", "q1", "", status="failed", raw={"error": "connection to db-node-1 timed out"}),
            case("b", "q2", "", status="failed", raw={"error": "connection to db-node-2 timed out"}),
        ),
    )
    output = tmp_path / "candidates.yaml"

    args = parse(
        [
            "--from-failures",
            "--baseline",
            str(baseline_path),
            "--current",
            str(current_path),
            "--output",
            str(output),
            "--similarity-threshold",
            "0.99",
        ]
    )
    assert _cmd_generate_cases(args) == 0
    text = output.read_text(encoding="utf-8")
    # At a strict 0.99 threshold the two near-duplicate failures should NOT merge.
    assert "a__regression" in text
    assert "b__regression" in text
