import json
from pathlib import Path

import pytest

from agenteval.cli import _cmd_generate_cases, build_parser
from agenteval.core.generator import (
    generate_cases_from_logs,
    propose_cases_from_jsonl,
    propose_cases_from_run_report,
)
from agenteval.core.schema import load_test_cases


def run_report(*case_results):
    return {
        "run_id": "r1",
        "timestamp": "2026-01-01T00:00:00Z",
        "case_results": list(case_results),
    }


def case(case_id, prompt, final_answer, *, status="passed", **overrides):
    return {
        "case_id": case_id,
        "prompt": prompt,
        "final_answer": final_answer,
        "status": status,
        **overrides,
    }


# --- propose_cases_from_run_report ------------------------------------------------


def test_propose_from_run_report_produces_candidate_cases():
    report = run_report(case("c1", "How many customers?", "7043"))
    cases = propose_cases_from_run_report(report)
    assert len(cases) == 1
    proposed = cases[0]
    assert proposed.id == "c1__from_log"
    assert proposed.prompt == "How many customers?"
    assert proposed.expects.ground_truth == "7043"
    assert proposed.expects.correctness_type.value == "exact"
    assert proposed.source == "production_log"
    assert proposed.review_status == "candidate"
    assert "candidate" in proposed.tags


def test_propose_from_run_report_skips_agent_and_evaluator_errors():
    report = run_report(
        case("ok", "q1", "a1"),
        case("bad", "q2", "", status="agent_error"),
        case("judge_bad", "q3", "a3", status="evaluator_error"),
    )
    cases = propose_cases_from_run_report(report)
    assert [c.prompt for c in cases] == ["q1"]


def test_propose_from_run_report_skips_missing_status_derived_cases():
    # No explicit status and no correctness_pass -> derives to "skipped" via case_status()
    report = run_report({"case_id": "s1", "prompt": "q", "final_answer": "a"})
    cases = propose_cases_from_run_report(report)
    assert cases == []


def test_propose_from_run_report_dedupes_case_insensitive_whitespace_normalized():
    report = run_report(
        case("c1", "How many customers?", "7043"),
        case("c2", "  HOW  many CUSTOMERS?  ", "different answer"),
    )
    cases = propose_cases_from_run_report(report)
    assert len(cases) == 1


def test_propose_from_run_report_respects_limit():
    report = run_report(
        case("c1", "q1", "a1"),
        case("c2", "q2", "a2"),
        case("c3", "q3", "a3"),
    )
    cases = propose_cases_from_run_report(report, limit=2)
    assert len(cases) == 2


def test_propose_from_run_report_skips_blank_prompt_or_answer():
    report = run_report(case("c1", "", "a1"), case("c2", "q2", ""))
    assert propose_cases_from_run_report(report) == []


def test_propose_from_run_report_custom_correctness_type():
    report = run_report(case("c1", "how many", "42"))
    cases = propose_cases_from_run_report(report, correctness_type="numeric")
    assert cases[0].expects.correctness_type.value == "numeric"


# --- propose_cases_from_jsonl ------------------------------------------------------


def test_propose_from_jsonl_produces_candidate_cases(tmp_path):
    path = tmp_path / "logs.jsonl"
    path.write_text(
        '{"prompt": "q1", "answer": "a1"}\n{"prompt": "q2", "answer": "a2"}\n',
        encoding="utf-8",
    )
    cases = propose_cases_from_jsonl(path)
    assert len(cases) == 2
    assert cases[0].source == "production_log"
    assert cases[0].review_status == "candidate"


def test_propose_from_jsonl_skips_malformed_lines(tmp_path, capsys):
    path = tmp_path / "logs.jsonl"
    path.write_text(
        '{"prompt": "q1", "answer": "a1"}\nnot json at all\n["also", "not", "an", "object"]\n',
        encoding="utf-8",
    )
    cases = propose_cases_from_jsonl(path)
    assert len(cases) == 1
    err = capsys.readouterr().err
    assert "malformed JSONL line 2" in err
    assert "non-object JSONL line 3" in err


def test_propose_from_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "logs.jsonl"
    path.write_text('{"prompt": "q1", "answer": "a1"}\n\n\n', encoding="utf-8")
    assert len(propose_cases_from_jsonl(path)) == 1


def test_propose_from_jsonl_skips_missing_prompt_or_answer(tmp_path, capsys):
    path = tmp_path / "logs.jsonl"
    path.write_text('{"prompt": "q1"}\n{"answer": "a1"}\n', encoding="utf-8")
    assert propose_cases_from_jsonl(path) == []
    assert "missing prompt or answer" in capsys.readouterr().err


def test_propose_from_jsonl_dedupes_and_respects_limit(tmp_path):
    path = tmp_path / "logs.jsonl"
    lines = "\n".join(json.dumps({"prompt": f"q{i}", "answer": f"a{i}"}) for i in range(5))
    path.write_text(lines + "\n", encoding="utf-8")
    cases = propose_cases_from_jsonl(path, limit=3)
    assert len(cases) == 3


# --- generate_cases_from_logs (dispatcher) -----------------------------------------


def test_generate_cases_from_logs_run_report_format(tmp_path):
    report_path = tmp_path / "run.json"
    report_path.write_text(json.dumps(run_report(case("c1", "q1", "a1"))), encoding="utf-8")
    cases = generate_cases_from_logs(report_path, log_format="run-report")
    assert len(cases) == 1


def test_generate_cases_from_logs_jsonl_format(tmp_path):
    path = tmp_path / "logs.jsonl"
    path.write_text('{"prompt": "q1", "answer": "a1"}\n', encoding="utf-8")
    cases = generate_cases_from_logs(path, log_format="jsonl")
    assert len(cases) == 1


def test_generate_cases_from_logs_missing_file_raises(tmp_path):
    with pytest.raises(ValueError, match="logs file not found"):
        generate_cases_from_logs(tmp_path / "nope.json", log_format="run-report")


def test_generate_cases_from_logs_invalid_json_raises(tmp_path):
    path = tmp_path / "run.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        generate_cases_from_logs(path, log_format="run-report")


def test_generate_cases_from_logs_non_object_run_report_raises(tmp_path):
    path = tmp_path / "run.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        generate_cases_from_logs(path, log_format="run-report")


def test_generate_cases_from_logs_unknown_format_raises(tmp_path):
    path = tmp_path / "run.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown log_format"):
        generate_cases_from_logs(path, log_format="xml")


def test_generate_cases_from_logs_all_errors_raises_clear_message(tmp_path):
    report_path = tmp_path / "run.json"
    report_path.write_text(
        json.dumps(run_report(case("c1", "q1", "", status="agent_error"))), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="no usable cases found"):
        generate_cases_from_logs(report_path, log_format="run-report")


def test_generate_cases_from_logs_rejects_non_positive_limit(tmp_path):
    report_path = tmp_path / "run.json"
    report_path.write_text(json.dumps(run_report(case("c1", "q1", "a1"))), encoding="utf-8")
    with pytest.raises(ValueError, match="limit must be at least 1"):
        generate_cases_from_logs(report_path, log_format="run-report", limit=0)


# --- write_candidate_yaml round trip (reused writer, sanity check here) -----------


def test_proposed_cases_round_trip_through_write_candidate_yaml(tmp_path):
    from agenteval.core.generator import write_candidate_yaml

    report_path = tmp_path / "run.json"
    report_path.write_text(
        json.dumps(run_report(case("c1", "How many?", "42"))), encoding="utf-8"
    )
    cases = generate_cases_from_logs(report_path, log_format="run-report")
    out_path = write_candidate_yaml(cases, tmp_path / "candidates.yaml")
    loaded = load_test_cases(out_path)
    assert loaded[0].review_status == "candidate"
    assert loaded[0].source == "production_log"


# --- CLI --------------------------------------------------------------------------


def test_cli_generate_cases_end_to_end(tmp_path, capsys):
    report_path = tmp_path / "run.json"
    report_path.write_text(
        json.dumps(run_report(case("c1", "How many?", "42"))), encoding="utf-8"
    )
    output_path = tmp_path / "candidates.yaml"
    args = build_parser().parse_args(
        [
            "generate-cases",
            "--logs",
            str(report_path),
            "--output",
            str(output_path),
        ]
    )
    assert _cmd_generate_cases(args) == 0
    out = capsys.readouterr().out
    assert "proposed=1" in out
    assert "review_status=candidate" in out
    assert output_path.is_file()


def test_cli_generate_cases_jsonl_format(tmp_path, capsys):
    logs_path = tmp_path / "logs.jsonl"
    logs_path.write_text('{"prompt": "q1", "answer": "a1"}\n', encoding="utf-8")
    output_path = tmp_path / "candidates.yaml"
    args = build_parser().parse_args(
        [
            "generate-cases",
            "--logs",
            str(logs_path),
            "--format",
            "jsonl",
            "--output",
            str(output_path),
        ]
    )
    assert _cmd_generate_cases(args) == 0
    assert "proposed=1" in capsys.readouterr().out


def test_cli_generate_cases_output_collision_requires_overwrite(tmp_path, capsys):
    report_path = tmp_path / "run.json"
    report_path.write_text(
        json.dumps(run_report(case("c1", "How many?", "42"))), encoding="utf-8"
    )
    output_path = tmp_path / "candidates.yaml"
    output_path.write_text("existing", encoding="utf-8")
    args = build_parser().parse_args(
        ["generate-cases", "--logs", str(report_path), "--output", str(output_path)]
    )
    assert _cmd_generate_cases(args) == 2
    assert "output exists" in capsys.readouterr().err

    args_force = build_parser().parse_args(
        [
            "generate-cases",
            "--logs",
            str(report_path),
            "--output",
            str(output_path),
            "--overwrite",
        ]
    )
    assert _cmd_generate_cases(args_force) == 0


def test_cli_generate_cases_reports_clear_error_for_missing_logs(tmp_path, capsys):
    args = build_parser().parse_args(
        ["generate-cases", "--logs", str(tmp_path / "nope.json")]
    )
    assert _cmd_generate_cases(args) == 2
    assert "logs file not found" in capsys.readouterr().err
