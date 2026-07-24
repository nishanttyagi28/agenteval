"""Tests for the additive trajectory diff feature (core + CLI)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agenteval.cli import _cmd_diff, build_parser
from agenteval.core.trajectory_diff import (
    TrajectoryDiffError,
    TrajectorySide,
    TrajectoryStep,
    diff_trajectories,
    format_trajectory_diff,
    load_trajectory_file,
    parse_trajectory_payload,
)


def side_from_labels(*labels: str, score: float | None = None) -> TrajectorySide:
    return TrajectorySide(
        steps=tuple(TrajectoryStep(label=label) for label in labels),
        score=score,
    )


def write_json(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ── core logic ───────────────────────────────────────────────────────────────


def test_identical_trajectories_report_no_diff():
    result = diff_trajectories(
        side_from_labels("route:sql", "agent:sql", score=1.0),
        side_from_labels("route:sql", "agent:sql", score=1.0),
    )
    assert result.identical
    assert result.added == 0
    assert result.removed == 0
    assert result.changed == 0
    assert result.unchanged == 2
    assert result.similarity == 1.0
    assert result.score_delta == 0.0
    text = format_trajectory_diff(result)
    assert "identical" in text
    assert "0 added, 0 removed, 0 changed, 2 unchanged" in text


def test_completely_different_trajectories_are_all_added_and_removed():
    result = diff_trajectories(
        side_from_labels("route:sql", "agent:sql"),
        side_from_labels("route:ml", "agent:ml", "tool:report"),
    )
    assert not result.identical
    assert result.unchanged == 0
    assert result.removed == 2
    assert result.added == 3
    assert result.changed == 0
    assert result.similarity == 0.0
    kinds = [e.kind for e in result.entries]
    assert kinds.count("removed") == 2
    assert kinds.count("added") == 3


def test_empty_trajectory_on_either_side():
    both_empty = diff_trajectories(side_from_labels(), side_from_labels())
    assert both_empty.identical
    assert both_empty.similarity == 1.0
    assert "both trajectories are empty" in format_trajectory_diff(both_empty)

    only_a = diff_trajectories(side_from_labels("a", "b"), side_from_labels())
    assert only_a.removed == 2
    assert only_a.added == 0
    assert only_a.similarity == 0.0

    only_b = diff_trajectories(side_from_labels(), side_from_labels("x"))
    assert only_b.added == 1
    assert only_b.removed == 0


def test_different_lengths_with_shared_prefix_and_suffix():
    result = diff_trajectories(
        side_from_labels("route:sql", "agent:sql", "tool:summarize"),
        side_from_labels("route:sql", "agent:ml", "tool:summarize", "tool:report"),
        path_a="a.json",
        path_b="b.json",
    )
    assert result.unchanged == 2  # route:sql, tool:summarize
    assert result.removed == 1  # agent:sql
    assert result.added == 2  # agent:ml, tool:report
    assert result.changed == 0
    assert 0.0 < result.similarity < 1.0
    text = format_trajectory_diff(result)
    assert "- " in text and "agent:sql" in text
    assert "+ " in text and "agent:ml" in text
    assert "tool:report" in text
    assert "Summary:" in text


def test_changed_steps_detect_payload_differences():
    side_a = TrajectorySide(
        steps=(
            TrajectoryStep(label="tool:lookup", tool_calls=("lookup_tool",), output="Paris"),
            TrajectoryStep(label="agent:mock"),
        ),
        score=1.0,
    )
    side_b = TrajectorySide(
        steps=(
            TrajectoryStep(label="tool:lookup", tool_calls=("lookup_tool",), output="Lyon"),
            TrajectoryStep(label="agent:mock"),
        ),
        score=0.85,
    )
    result = diff_trajectories(side_a, side_b)
    assert result.changed == 1
    assert result.unchanged == 1
    assert result.added == 0
    assert result.removed == 0
    assert result.score_delta == pytest.approx(-0.15)
    changed = next(e for e in result.entries if e.kind == "changed")
    assert "output" in (changed.detail or "")
    text = format_trajectory_diff(result)
    assert "~ " in text
    assert "score delta: -0.15" in text


def test_parse_trajectory_evaluation_and_nodes_fired_shapes():
    side = parse_trajectory_payload(
        {"actual": ["route:sql", "agent:sql"], "score": 0.9, "exact_match": False}
    )
    assert side.labels == ("route:sql", "agent:sql")
    assert side.score == 0.9
    assert side.source_kind == "actual"

    case = parse_trajectory_payload(
        {
            "case_id": "c1",
            "nodes_fired": ["route:sql", "agent:sql"],
            "trajectory": {"actual": ["route:sql", "agent:sql"], "score": 1.0},
        }
    )
    # trajectory.actual takes precedence when present
    assert case.labels == ("route:sql", "agent:sql")
    assert case.score == 1.0


def test_load_file_and_run_report_case_id(tmp_path):
    path = write_json(
        tmp_path / "run.json",
        {
            "run_id": "r1",
            "case_results": [
                {"case_id": "other", "nodes_fired": ["a"]},
                {
                    "case_id": "target",
                    "nodes_fired": ["route:sql", "agent:sql"],
                    "trajectory": {"score": 0.75},
                },
            ],
        },
    )
    side = load_trajectory_file(path, case_id="target")
    assert side.labels == ("route:sql", "agent:sql")
    assert side.score == 0.75

    with pytest.raises(TrajectoryDiffError, match="case_id"):
        load_trajectory_file(path, case_id="missing")

    with pytest.raises(TrajectoryDiffError, match="multiple cases"):
        load_trajectory_file(path)


def test_json_and_text_renderers_agree_on_summary():
    result = diff_trajectories(
        side_from_labels("a", "b", score=1.0),
        side_from_labels("a", "c", score=0.5),
    )
    payload = result.to_dict()
    assert payload["summary"]["added"] == result.added == 1
    assert payload["summary"]["removed"] == result.removed == 1
    assert payload["summary"]["unchanged"] == 1
    assert payload["summary"]["score_delta"] == pytest.approx(-0.5)
    assert "entries" in payload
    text = format_trajectory_diff(result, verbose=True)
    assert "= [0/0] a" in text
    assert "1 added, 1 removed" in text


# ── CLI ──────────────────────────────────────────────────────────────────────


def test_cli_diff_text_and_json(tmp_path, capsys):
    a = write_json(tmp_path / "a.json", ["route:sql", "agent:sql"])
    b = write_json(tmp_path / "b.json", ["route:sql", "agent:ml", "tool:report"])

    args = build_parser().parse_args(["diff", str(a), str(b)])
    assert _cmd_diff(args) == 0
    out = capsys.readouterr().out
    assert "Trajectory diff" in out
    assert "Summary:" in out
    assert "agent:sql" in out
    assert "agent:ml" in out

    args_json = build_parser().parse_args(["diff", str(a), str(b), "--json"])
    assert _cmd_diff(args_json) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["removed"] == 1
    assert payload["summary"]["added"] == 2
    assert payload["identical"] is False


def test_cli_diff_identical_sample_shape(tmp_path, capsys):
    payload = {"actual": ["route:sql", "agent:sql"], "score": 1.0}
    a = write_json(tmp_path / "a.json", payload)
    b = write_json(tmp_path / "b.json", payload)
    assert _cmd_diff(build_parser().parse_args(["diff", str(a), str(b)])) == 0
    out = capsys.readouterr().out
    assert "identical" in out
    assert "0 added, 0 removed, 0 changed" in out


def test_cli_diff_missing_file_is_clean_error(tmp_path, capsys):
    a = write_json(tmp_path / "a.json", ["x"])
    args = build_parser().parse_args(
        ["diff", str(a), str(tmp_path / "missing.json")]
    )
    assert _cmd_diff(args) == 2
    assert "error:" in capsys.readouterr().err


def test_cli_diff_invalid_json_is_clean_error(tmp_path, capsys):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text("{not-json", encoding="utf-8")
    b.write_text("[]", encoding="utf-8")
    assert _cmd_diff(build_parser().parse_args(["diff", str(a), str(b)])) == 2
    assert "error:" in capsys.readouterr().err
