"""Step-by-step diff of two agent trajectories.

Pure, additive utility: does not score cases, does not participate in the
regression gate, and does not alter ``evaluate_trajectory``. It reuses the
same ordered step-label model (``nodes_fired`` / ``trajectory.actual``) and
optional structured fields from ``trace_steps`` (name, input, output).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from agenteval.core.trajectory import _lcs_pairs, _normalize_steps


class TrajectoryDiffError(ValueError):
    """Raised when a trajectory payload cannot be loaded or interpreted."""


@dataclass(frozen=True)
class TrajectoryStep:
    """One normalized step drawn from trajectory / nodes_fired / trace data."""

    label: str
    tool_calls: tuple[str, ...] = ()
    input: Any = None
    output: Any = None

    def payload_equal(self, other: TrajectoryStep) -> bool:
        return (
            self.label == other.label
            and self.tool_calls == other.tool_calls
            and self.input == other.input
            and self.output == other.output
        )


@dataclass(frozen=True)
class DiffEntry:
    """One aligned edit between trajectory A and trajectory B."""

    kind: str  # unchanged | added | removed | changed
    index_a: int | None
    index_b: int | None
    step_a: str | None
    step_b: str | None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrajectorySide:
    """Loaded content for one side of the diff."""

    steps: tuple[TrajectoryStep, ...]
    score: float | None = None
    source_kind: str = "steps"

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(step.label for step in self.steps)


@dataclass
class TrajectoryDiffResult:
    """Full diff result for text or JSON rendering."""

    entries: list[DiffEntry] = field(default_factory=list)
    steps_a: list[str] = field(default_factory=list)
    steps_b: list[str] = field(default_factory=list)
    added: int = 0
    removed: int = 0
    changed: int = 0
    unchanged: int = 0
    similarity: float = 0.0
    score_a: float | None = None
    score_b: float | None = None
    score_delta: float | None = None
    identical: bool = False
    path_a: str = ""
    path_b: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path_a": self.path_a,
            "path_b": self.path_b,
            "steps_a": list(self.steps_a),
            "steps_b": list(self.steps_b),
            "identical": self.identical,
            "summary": {
                "added": self.added,
                "removed": self.removed,
                "changed": self.changed,
                "unchanged": self.unchanged,
                "similarity": self.similarity,
                "score_a": self.score_a,
                "score_b": self.score_b,
                "score_delta": self.score_delta,
            },
            "entries": [entry.to_dict() for entry in self.entries],
        }


def _as_step_label(item: Any, index: int) -> str:
    if isinstance(item, str):
        label = item.strip()
        if not label:
            raise TrajectoryDiffError(f"steps[{index}] must not be blank")
        return label
    if isinstance(item, dict):
        if "name" in item and item["name"] is not None:
            name = str(item["name"]).strip()
            kind = str(item.get("kind") or "").strip()
            if kind and name:
                return f"{kind}:{name}" if not name.startswith(f"{kind}:") else name
            if name:
                return name
        if "label" in item and item["label"] is not None:
            label = str(item["label"]).strip()
            if label:
                return label
        raise TrajectoryDiffError(
            f"steps[{index}] mapping must include a non-empty 'name' or 'label'"
        )
    raise TrajectoryDiffError(f"steps[{index}] must be a string or mapping")


def _step_from_item(item: Any, index: int) -> TrajectoryStep:
    if isinstance(item, str):
        return TrajectoryStep(label=_as_step_label(item, index))
    if not isinstance(item, dict):
        raise TrajectoryDiffError(f"steps[{index}] must be a string or mapping")
    label = _as_step_label(item, index)
    tools_raw = item.get("tool_calls") or item.get("tools_called") or []
    if tools_raw is None:
        tools_raw = []
    if not isinstance(tools_raw, (list, tuple)):
        raise TrajectoryDiffError(f"steps[{index}].tool_calls must be a list")
    tools = tuple(str(t) for t in tools_raw)
    return TrajectoryStep(
        label=label,
        tool_calls=tools,
        input=item.get("input"),
        output=item.get("output"),
    )


def _steps_from_sequence(raw: Sequence[Any], *, label: str) -> tuple[TrajectoryStep, ...]:
    if not isinstance(raw, (list, tuple)):
        raise TrajectoryDiffError(f"{label} must be a list of steps")
    return tuple(_step_from_item(item, index) for index, item in enumerate(raw))


def _score_from_mapping(data: dict[str, Any]) -> float | None:
    for key in ("score", "trajectory_score", "f1"):
        if key in data and data[key] is not None:
            try:
                return float(data[key])
            except (TypeError, ValueError) as exc:
                raise TrajectoryDiffError(f"{key} must be a number") from exc
    trajectory = data.get("trajectory")
    if isinstance(trajectory, dict) and trajectory.get("score") is not None:
        try:
            return float(trajectory["score"])
        except (TypeError, ValueError) as exc:
            raise TrajectoryDiffError("trajectory.score must be a number") from exc
    return None


def parse_trajectory_payload(
    data: Any,
    *,
    case_id: str | None = None,
) -> TrajectorySide:
    """Normalize a JSON payload into ordered steps + optional score.

    Accepted shapes (trajectory scoring / run-report conventions):
    - list of step strings (``nodes_fired`` / ``trajectory.actual``)
    - list of step mappings (``trace_steps``-like: name/kind/input/output)
    - ``TrajectoryEvaluation`` mapping with ``actual`` (+ optional ``score``)
    - case result with ``nodes_fired`` and/or ``trajectory`` / ``trace_steps``
    - ``{"steps": [...]}`` explicit wrapper
    - full run report with ``case_results`` (optional ``case_id``)
    """
    if isinstance(data, list):
        return TrajectorySide(steps=_steps_from_sequence(data, label="trajectory"), source_kind="list")

    if not isinstance(data, dict):
        raise TrajectoryDiffError("trajectory payload must be a JSON list or object")

    if "case_results" in data:
        cases = data.get("case_results") or []
        if not isinstance(cases, list) or not cases:
            raise TrajectoryDiffError("run report has no case_results to diff")
        selected: dict[str, Any] | None = None
        if case_id is not None:
            for case in cases:
                if isinstance(case, dict) and case.get("case_id") == case_id:
                    selected = case
                    break
            if selected is None:
                available = ", ".join(
                    sorted(str(c.get("case_id") or "?") for c in cases if isinstance(c, dict))
                )
                raise TrajectoryDiffError(
                    f"case_id {case_id!r} not found in run report. Available: {available}"
                )
        elif len(cases) == 1 and isinstance(cases[0], dict):
            selected = cases[0]
        else:
            raise TrajectoryDiffError(
                "run report has multiple cases; pass --case-id to select one"
            )
        return parse_trajectory_payload(selected, case_id=None)

    score = _score_from_mapping(data)

    if "steps" in data:
        return TrajectorySide(
            steps=_steps_from_sequence(data["steps"], label="steps"),
            score=score,
            source_kind="steps",
        )

    if "trace_steps" in data and data["trace_steps"] is not None:
        return TrajectorySide(
            steps=_steps_from_sequence(data["trace_steps"], label="trace_steps"),
            score=score,
            source_kind="trace_steps",
        )

    trajectory = data.get("trajectory")
    if isinstance(trajectory, dict) and trajectory.get("actual") is not None:
        return TrajectorySide(
            steps=_steps_from_sequence(trajectory["actual"], label="trajectory.actual"),
            score=score if score is not None else _score_from_mapping(trajectory),
            source_kind="trajectory.actual",
        )

    if data.get("actual") is not None:
        return TrajectorySide(
            steps=_steps_from_sequence(data["actual"], label="actual"),
            score=score,
            source_kind="actual",
        )

    if data.get("nodes_fired") is not None:
        return TrajectorySide(
            steps=_steps_from_sequence(data["nodes_fired"], label="nodes_fired"),
            score=score,
            source_kind="nodes_fired",
        )

    raise TrajectoryDiffError(
        "unrecognized trajectory shape; expected a step list, "
        "TrajectoryEvaluation (actual/score), nodes_fired, trace_steps, or a run report"
    )


def load_trajectory_file(
    path: str | Path,
    *,
    case_id: str | None = None,
) -> TrajectorySide:
    """Load and normalize one trajectory JSON file."""
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise TrajectoryDiffError(f"cannot read {source}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TrajectoryDiffError(f"invalid JSON in {source}: {exc}") from exc
    return parse_trajectory_payload(data, case_id=case_id)


def _describe_change(step_a: TrajectoryStep, step_b: TrajectoryStep) -> str:
    parts: list[str] = []
    if step_a.label != step_b.label:
        parts.append(f"label: {step_a.label!r} -> {step_b.label!r}")
    if step_a.tool_calls != step_b.tool_calls:
        parts.append(
            f"tool_calls: {list(step_a.tool_calls)!r} -> {list(step_b.tool_calls)!r}"
        )
    if step_a.input != step_b.input:
        parts.append(f"input: {step_a.input!r} -> {step_b.input!r}")
    if step_a.output != step_b.output:
        parts.append(f"output: {step_a.output!r} -> {step_b.output!r}")
    return "; ".join(parts) if parts else "payload differs"


def _similarity(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    """LCS F1 between two step-label sequences (empty/empty → 1.0)."""
    if not labels_a and not labels_b:
        return 1.0
    if not labels_a or not labels_b:
        return 0.0
    # Reuse trajectory normalization + LCS (same algorithm as evaluate_trajectory).
    a = _normalize_steps(list(labels_a), label="steps_a")
    b = _normalize_steps(list(labels_b), label="steps_b")
    pairs = _lcs_pairs(a, b)
    matched = len(pairs)
    precision = matched / len(b) if b else 0.0
    recall = matched / len(a) if a else 0.0
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def diff_trajectories(
    side_a: TrajectorySide,
    side_b: TrajectorySide,
    *,
    path_a: str = "",
    path_b: str = "",
) -> TrajectoryDiffResult:
    """Compute an ordered step-by-step diff between two loaded trajectories."""
    labels_a = side_a.labels
    labels_b = side_b.labels
    # Empty labels are allowed (edge case: empty trajectory).
    pairs = ()
    if labels_a and labels_b:
        a_norm = _normalize_steps(list(labels_a), label="steps_a")
        b_norm = _normalize_steps(list(labels_b), label="steps_b")
        # Labels are already stripped; normalize is identity for non-blank.
        pairs = _lcs_pairs(a_norm, b_norm)
    elif not labels_a and not labels_b:
        pairs = ()

    matched_a = {i for i, _ in pairs}
    matched_b = {j for _, j in pairs}
    pair_map = {i: j for i, j in pairs}

    entries: list[DiffEntry] = []
    i = 0
    j = 0
    n_a = len(side_a.steps)
    n_b = len(side_b.steps)

    while i < n_a or j < n_b:
        if i < n_a and i not in matched_a:
            entries.append(
                DiffEntry(
                    kind="removed",
                    index_a=i,
                    index_b=None,
                    step_a=side_a.steps[i].label,
                    step_b=None,
                )
            )
            i += 1
            continue
        if j < n_b and j not in matched_b:
            entries.append(
                DiffEntry(
                    kind="added",
                    index_a=None,
                    index_b=j,
                    step_a=None,
                    step_b=side_b.steps[j].label,
                )
            )
            j += 1
            continue
        # Both sides should now be an LCS-matched pair.
        if i < n_a and j < n_b and pair_map.get(i) == j:
            step_a = side_a.steps[i]
            step_b = side_b.steps[j]
            if step_a.payload_equal(step_b):
                entries.append(
                    DiffEntry(
                        kind="unchanged",
                        index_a=i,
                        index_b=j,
                        step_a=step_a.label,
                        step_b=step_b.label,
                    )
                )
            else:
                entries.append(
                    DiffEntry(
                        kind="changed",
                        index_a=i,
                        index_b=j,
                        step_a=step_a.label,
                        step_b=step_b.label,
                        detail=_describe_change(step_a, step_b),
                    )
                )
            i += 1
            j += 1
            continue
        # Safety: advance the side that is lagging if pairing is inconsistent.
        if i < n_a:
            entries.append(
                DiffEntry(
                    kind="removed",
                    index_a=i,
                    index_b=None,
                    step_a=side_a.steps[i].label,
                    step_b=None,
                )
            )
            i += 1
        elif j < n_b:
            entries.append(
                DiffEntry(
                    kind="added",
                    index_a=None,
                    index_b=j,
                    step_a=None,
                    step_b=side_b.steps[j].label,
                )
            )
            j += 1

    added = sum(1 for e in entries if e.kind == "added")
    removed = sum(1 for e in entries if e.kind == "removed")
    changed = sum(1 for e in entries if e.kind == "changed")
    unchanged = sum(1 for e in entries if e.kind == "unchanged")
    score_a = side_a.score
    score_b = side_b.score
    score_delta = None
    if score_a is not None and score_b is not None:
        score_delta = score_b - score_a
    identical = added == 0 and removed == 0 and changed == 0

    return TrajectoryDiffResult(
        entries=entries,
        steps_a=list(labels_a),
        steps_b=list(labels_b),
        added=added,
        removed=removed,
        changed=changed,
        unchanged=unchanged,
        similarity=_similarity(labels_a, labels_b),
        score_a=score_a,
        score_b=score_b,
        score_delta=score_delta,
        identical=identical,
        path_a=path_a,
        path_b=path_b,
    )


def format_trajectory_diff(result: TrajectoryDiffResult, *, verbose: bool = False) -> str:
    """Human-readable terminal output (plain text; CLI does not use ANSI colors)."""
    lines = [
        "=== Trajectory diff ===",
        f"a: {result.path_a or '(a)'} ({len(result.steps_a)} steps)",
        f"b: {result.path_b or '(b)'} ({len(result.steps_b)} steps)",
        "",
    ]

    if result.identical and not result.steps_a and not result.steps_b:
        lines.append("(both trajectories are empty)")
    elif result.identical:
        lines.append(f"= {result.unchanged} step(s) unchanged — trajectories are identical")
    else:
        unchanged_buffer = 0

        def flush_unchanged() -> None:
            nonlocal unchanged_buffer
            if unchanged_buffer:
                if verbose:
                    pass  # already printed individually
                else:
                    lines.append(f"= {unchanged_buffer} step(s) unchanged")
                unchanged_buffer = 0

        for entry in result.entries:
            if entry.kind == "unchanged":
                if verbose:
                    lines.append(
                        f"= [{entry.index_a}/{entry.index_b}] {entry.step_a}"
                    )
                else:
                    unchanged_buffer += 1
                continue
            flush_unchanged()
            if entry.kind == "removed":
                lines.append(f"- [{entry.index_a}/-] {entry.step_a}")
            elif entry.kind == "added":
                lines.append(f"+ [-/{entry.index_b}] {entry.step_b}")
            elif entry.kind == "changed":
                if entry.step_a == entry.step_b:
                    lines.append(f"~ [{entry.index_a}/{entry.index_b}] {entry.step_a}")
                else:
                    lines.append(
                        f"~ [{entry.index_a}/{entry.index_b}] "
                        f"{entry.step_a} -> {entry.step_b}"
                    )
                if entry.detail:
                    lines.append(f"    {entry.detail}")
        flush_unchanged()

    lines.append("")
    summary = (
        f"Summary: {result.added} added, {result.removed} removed, "
        f"{result.changed} changed, {result.unchanged} unchanged"
    )
    lines.append(summary)
    lines.append(f"similarity (LCS F1): {result.similarity:.2f}")
    if result.score_delta is not None:
        lines.append(
            f"score delta: {result.score_delta:+.2f} "
            f"(a={result.score_a:.2f} -> b={result.score_b:.2f})"
        )
    elif result.score_a is not None or result.score_b is not None:
        lines.append(
            f"score: a={_fmt_score(result.score_a)} b={_fmt_score(result.score_b)}"
        )
    return "\n".join(lines) + "\n"


def _fmt_score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"
