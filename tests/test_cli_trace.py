import json

import pytest

from agenteval.cli import _cmd_trace, build_parser


def write_run(tmp_path, name="run.json", cases=None):
    cases = cases if cases is not None else [
        {
            "case_id": "c1",
            "status": "passed",
            "prompt": "2+2?",
            "trace_steps": [
                {"step_index": 0, "kind": "tool_call", "name": "calculator", "input": "2+2", "output": "4"}
            ],
        }
    ]
    path = tmp_path / name
    path.write_text(json.dumps({"run_id": "r1", "case_results": cases}), encoding="utf-8")
    return path


def parse(argv):
    return build_parser().parse_args(["trace", *argv])


def test_trace_prints_text_replay_by_default(tmp_path, capsys):
    run_path = write_run(tmp_path)

    args = parse([str(run_path), "--case-id", "c1"])
    assert _cmd_trace(args) == 0

    out = capsys.readouterr().out
    assert "case_id: c1" in out
    assert "calculator" in out


def test_trace_writes_html_when_requested(tmp_path):
    run_path = write_run(tmp_path)
    html_path = tmp_path / "trace.html"

    args = parse([str(run_path), "--case-id", "c1", "--html", str(html_path)])
    assert _cmd_trace(args) == 0

    assert html_path.is_file()
    assert "<!doctype html>" in html_path.read_text(encoding="utf-8")


def test_trace_unknown_case_id_is_a_clean_error(tmp_path, capsys):
    run_path = write_run(tmp_path)

    args = parse([str(run_path), "--case-id", "does_not_exist"])
    assert _cmd_trace(args) == 2
    err = capsys.readouterr().err
    assert "error:" in err
    assert "does_not_exist" in err


def test_trace_missing_run_file_is_a_clean_error(tmp_path, capsys):
    args = parse([str(tmp_path / "nope.json"), "--case-id", "c1"])
    assert _cmd_trace(args) == 2
    assert "error:" in capsys.readouterr().err


def test_trace_corrupted_run_json_is_a_clean_error(tmp_path, capsys):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")

    args = parse([str(path), "--case-id", "c1"])
    assert _cmd_trace(args) == 2
    assert "error:" in capsys.readouterr().err


def test_case_id_argument_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["trace", "run.json"])
