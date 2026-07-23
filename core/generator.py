"""Generate reviewable candidate cases: adversarial mutations of hand-written
golden cases, and proposals mined from production run logs.

Both generators share one convention rather than inventing a second one:
every proposed case is written with ``review_status="candidate"`` (see
``TestCase``) and stays out of the blocking golden gate until a human
reviews and promotes it. ``write_candidate_yaml``/``test_case_to_dict``
below are already generic over any sequence of ``TestCase`` objects, so both
generators reuse the same writer.
"""

from __future__ import annotations

import difflib
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Sequence

from agenteval.core.compare import case_status
from agenteval.core.config import resolve_agent_repo
from agenteval.core.schema import CorrectnessType, Expects, TestCase

MUTATION_TYPES = {
    "reworded",
    "ambiguous_context",
    "distractor",
    "numeric_precision",
    "mild_injection",
}

ChatFunction = Callable[..., tuple[str | None, str | None]]


def _expects_dict(expects: Expects) -> dict[str, Any]:
    data = asdict(expects)
    correctness_type = data.get("correctness_type")
    if isinstance(correctness_type, CorrectnessType):
        data["correctness_type"] = correctness_type.value
    if data.get("evaluator") is None:
        data.pop("evaluator", None)
    return data


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    if not text or not text.strip():
        raise ValueError("generator returned an empty response")
    raw = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if fenced:
        raw = fenced.group(1).strip()
    start, end = raw.find("["), raw.rfind("]")
    if start < 0 or end < start:
        raise ValueError("generator response does not contain a JSON array")
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"generator returned invalid JSON: {exc}") from exc
    if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
        raise ValueError("generator JSON must be an array of objects")
    return parsed


def _default_chat() -> ChatFunction:
    agent_repo = resolve_agent_repo()
    if str(agent_repo) not in sys.path:
        sys.path.insert(0, str(agent_repo))
    from utils.env import load_project_env

    load_project_env()
    from agents.llm_client import chat_completion

    return chat_completion


def generator_messages(case: TestCase, variants: int) -> list[dict[str, str]]:
    allowed = ", ".join(sorted(MUTATION_TYPES))
    expectations = json.dumps(_expects_dict(case.expects), ensure_ascii=False)
    system = (
        "You generate adversarial test prompts for an AI data-analysis agent. "
        "Every variant MUST preserve the exact question meaning, dataset scope, ground truth, "
        "and expected tool route. Do not introduce facts that change the correct answer. "
        "Ambiguity may add distracting wording but must remain answerable. Mild injection may "
        "ask the agent to guess or skip tools, while the original correct answer remains valid. "
        "Return JSON only: an array of objects with mutation_type and prompt."
    )
    user = (
        f"Create exactly {variants} distinct variants.\n"
        f"Allowed mutation types: {allowed}\n"
        f"Original id: {case.id}\nOriginal prompt: {case.prompt}\n"
        f"Immutable expectations: {expectations}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_variants(
    case: TestCase,
    *,
    variants: int = 3,
    chat: ChatFunction | None = None,
) -> list[TestCase]:
    if variants < 1:
        raise ValueError("variants must be at least 1")
    chat_fn = chat or _default_chat()
    response, error = chat_fn(
        generator_messages(case, variants),
        temperature=0.4,
        max_tokens=max(700, variants * 220),
    )
    if error:
        raise RuntimeError(f"adversarial generator error for {case.id}: {error}")
    items = _extract_json_array(response or "")
    if len(items) != variants:
        raise ValueError(
            f"generator returned {len(items)} variants for {case.id}; expected {variants}"
        )

    generated: list[TestCase] = []
    seen_prompts = {case.prompt.strip().lower()}
    for index, item in enumerate(items, start=1):
        mutation = str(item.get("mutation_type") or "").strip().lower()
        prompt = str(item.get("prompt") or "").strip()
        if mutation not in MUTATION_TYPES:
            raise ValueError(f"unsupported mutation_type {mutation!r} for {case.id}")
        if not prompt:
            raise ValueError(f"empty adversarial prompt for {case.id}")
        normalized = re.sub(r"\s+", " ", prompt.lower())
        if normalized in seen_prompts:
            raise ValueError(f"duplicate adversarial prompt for {case.id}")
        seen_prompts.add(normalized)
        generated.append(
            TestCase(
                id=f"{case.id}__{mutation}__{index:02d}",
                prompt=prompt,
                expects=Expects.from_dict(_expects_dict(case.expects)),
                tags=list(dict.fromkeys([*case.tags, "adversarial", mutation])),
                source="adversarial",
                parent_id=case.id,
                mutation_type=mutation,
                review_status="candidate",
            )
        )
    return generated


def generate_suite(
    cases: Sequence[TestCase],
    *,
    variants_per_case: int = 3,
    chat: ChatFunction | None = None,
) -> list[TestCase]:
    generated: list[TestCase] = []
    for case in cases:
        generated.extend(generate_variants(case, variants=variants_per_case, chat=chat))
    return generated


def test_case_to_dict(case: TestCase) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": case.id,
        "prompt": case.prompt,
        "expects": _expects_dict(case.expects),
        "tags": case.tags,
    }
    for key in ("source", "parent_id", "mutation_type", "review_status"):
        value = getattr(case, key)
        if value is not None:
            data[key] = value
    return data


# ── generate-cases: propose candidates from production logs ─────────────────

_LOG_FORMATS = ("run-report", "jsonl")


def _log_candidate(
    prompt: str,
    answer: str,
    case_id: str,
    *,
    correctness_type: str,
) -> TestCase:
    return TestCase(
        id=case_id,
        prompt=prompt,
        expects=Expects(
            correctness_type=CorrectnessType(correctness_type),
            ground_truth=answer,
        ),
        tags=["candidate", "production_log"],
        source="production_log",
        review_status="candidate",
    )


def propose_cases_from_run_report(
    report: dict[str, Any],
    *,
    correctness_type: str = "exact",
    limit: int | None = None,
) -> list[TestCase]:
    """Propose candidate cases from an already-scored ``agenteval run`` report.

    Cases whose status is ``agent_error``/``evaluator_error``/``skipped``/
    ``missing`` are skipped — there is no reliable observed answer to seed a
    ground truth from. Duplicate prompts (case-insensitive, whitespace
    -normalized) are deduplicated, keeping the first occurrence.
    """
    cases: list[TestCase] = []
    seen_prompts: set[str] = set()
    for entry in report.get("case_results") or []:
        if not isinstance(entry, dict):
            continue
        if case_status(entry) in {"agent_error", "evaluator_error", "skipped", "missing"}:
            continue
        prompt = str(entry.get("prompt") or "").strip()
        answer = str(entry.get("final_answer") or "").strip()
        if not prompt or not answer:
            continue
        normalized = re.sub(r"\s+", " ", prompt.lower())
        if normalized in seen_prompts:
            continue
        seen_prompts.add(normalized)
        base_id = str(entry.get("case_id") or f"log_case_{len(cases) + 1}")
        cases.append(
            _log_candidate(
                prompt, answer, f"{base_id}__from_log", correctness_type=correctness_type
            )
        )
        if limit is not None and len(cases) >= limit:
            break
    return cases


def propose_cases_from_jsonl(
    path: str | Path,
    *,
    correctness_type: str = "exact",
    limit: int | None = None,
) -> list[TestCase]:
    """Propose candidate cases from a JSONL file of ``{"prompt": ..., "answer": ...}`` lines.

    Malformed or incomplete lines are skipped with a printed warning rather
    than aborting the whole batch — sample logs are rarely perfectly clean.
    """
    cases: list[TestCase] = []
    seen_prompts: set[str] = set()
    text = Path(path).read_text(encoding="utf-8")
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            print(f"warning: skipping malformed JSONL line {line_number}", file=sys.stderr)
            continue
        if not isinstance(record, dict):
            print(f"warning: skipping non-object JSONL line {line_number}", file=sys.stderr)
            continue
        prompt = str(record.get("prompt") or "").strip()
        answer = str(record.get("answer") or "").strip()
        if not prompt or not answer:
            print(
                f"warning: skipping JSONL line {line_number}: missing prompt or answer",
                file=sys.stderr,
            )
            continue
        normalized = re.sub(r"\s+", " ", prompt.lower())
        if normalized in seen_prompts:
            continue
        seen_prompts.add(normalized)
        cases.append(
            _log_candidate(
                prompt, answer, f"log_line_{line_number}__from_log", correctness_type=correctness_type
            )
        )
        if limit is not None and len(cases) >= limit:
            break
    return cases


def generate_cases_from_logs(
    logs_path: str | Path,
    *,
    log_format: str = "run-report",
    correctness_type: str = "exact",
    limit: int | None = None,
) -> list[TestCase]:
    """Propose candidate golden cases from a run-report JSON or a JSONL sample log."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    path = Path(logs_path)
    if not path.is_file():
        raise ValueError(f"logs file not found: {path}")
    if log_format == "run-report":
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"logs file is not valid JSON: {path}: {exc}") from exc
        if not isinstance(report, dict):
            raise ValueError(f"run-report logs must be a JSON object: {path}")
        cases = propose_cases_from_run_report(
            report, correctness_type=correctness_type, limit=limit
        )
    elif log_format == "jsonl":
        cases = propose_cases_from_jsonl(path, correctness_type=correctness_type, limit=limit)
    else:
        raise ValueError(
            f"unknown log_format {log_format!r}; expected one of {_LOG_FORMATS}"
        )
    if not cases:
        raise ValueError("no usable cases found in the provided logs")
    return cases


# ── generate-cases --from-failures: targeted regression candidates ─────────
#
# Unlike propose_cases_from_run_report above (which trusts any surviving
# final_answer as ground truth -- silently wrong for a case whose
# correctness_pass was False), this mines a *baseline vs current* pair: only
# cases where the baseline passed and the current run failed/errored
# qualify, so the baseline's own final_answer is trustworthy ground truth
# (it already passed correctness) rather than a guess.


def _failure_signature(entry: dict[str, Any]) -> str:
    """Normalized text used to detect near-duplicate failures for clustering."""
    error = (entry.get("raw") or {}).get("error")
    text = str(error or entry.get("final_answer") or entry.get("judge_reason") or "")
    return re.sub(r"\s+", " ", text.strip().lower())


def _cluster_failures(
    entries: Sequence[dict[str, Any]], *, similarity_threshold: float
) -> list[list[dict[str, Any]]]:
    """Greedy, deterministic near-duplicate grouping by failure-signature similarity.

    Each entry joins the first existing cluster whose representative (that
    cluster's first member) has a ``difflib.SequenceMatcher`` ratio at or
    above ``similarity_threshold`` against it; otherwise it starts a new
    cluster. First-seen-wins as representative, so results are reproducible
    given the same input order -- this is deliberately simple (no external
    clustering/NLP dependency) and easy to verify: the similarity metric is
    stdlib and its behavior on two strings is directly testable.
    """
    clusters: list[list[dict[str, Any]]] = []
    representative_signatures: list[str] = []
    for entry in entries:
        signature = _failure_signature(entry)
        placed = False
        for cluster, representative_signature in zip(clusters, representative_signatures):
            ratio = difflib.SequenceMatcher(None, signature, representative_signature).ratio()
            if ratio >= similarity_threshold:
                cluster.append(entry)
                placed = True
                break
        if not placed:
            clusters.append([entry])
            representative_signatures.append(signature)
    return clusters


def propose_regression_cases_from_failures(
    baseline: dict[str, Any],
    current: dict[str, Any],
    *,
    correctness_type: str = "exact",
    similarity_threshold: float = 0.85,
    limit: int | None = None,
) -> list[TestCase]:
    """Propose regression-test candidates from cases that regressed baseline -> current.

    Only a case where the baseline passed and the current run failed or
    errored qualifies. Near-duplicate failures (by normalized error/answer
    text) are clustered so one underlying regression doesn't produce many
    redundant candidates -- only each cluster's first (lowest case_id, for
    determinism) member becomes a candidate, tagged with its cluster size.
    The baseline's own recorded tool calls are carried over as
    ``must_call_tools`` so the generated case pins the tool route that used
    to work, not just the final answer text.
    """
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between 0 and 1")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")

    baseline_by_id = {
        str(entry["case_id"]): entry
        for entry in (baseline.get("case_results") or [])
        if isinstance(entry, dict) and entry.get("case_id")
    }
    current_by_id = {
        str(entry["case_id"]): entry
        for entry in (current.get("case_results") or [])
        if isinstance(entry, dict) and entry.get("case_id")
    }

    regressed_ids = sorted(
        case_id
        for case_id, current_entry in current_by_id.items()
        if case_id in baseline_by_id
        and case_status(baseline_by_id[case_id]) == "passed"
        and case_status(current_entry) in ("failed", "agent_error")
    )
    regressed_entries = [current_by_id[case_id] for case_id in regressed_ids]
    clusters = _cluster_failures(regressed_entries, similarity_threshold=similarity_threshold)

    cases: list[TestCase] = []
    for cluster in clusters:
        representative = cluster[0]
        case_id = str(representative["case_id"])
        baseline_entry = baseline_by_id[case_id]
        prompt = str(baseline_entry.get("prompt") or representative.get("prompt") or "").strip()
        ground_truth = str(baseline_entry.get("final_answer") or "").strip()
        if not prompt or not ground_truth:
            continue
        cases.append(
            TestCase(
                id=f"{case_id}__regression",
                prompt=prompt,
                expects=Expects(
                    correctness_type=CorrectnessType(correctness_type),
                    ground_truth=ground_truth,
                    must_call_tools=list(baseline_entry.get("tools_called") or []),
                ),
                tags=["candidate", "regression_from_failure", f"cluster_size:{len(cluster)}"],
                source="regression_from_failure",
                parent_id=case_id,
                review_status="candidate",
            )
        )
        if limit is not None and len(cases) >= limit:
            break
    if not cases:
        raise ValueError("no baseline-passed/current-failed regressions found between these two runs")
    return cases


def write_candidate_yaml(cases: Sequence[TestCase], path: str | Path) -> Path:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to write adversarial cases") from exc
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = [test_case_to_dict(case) for case in cases]
    output.write_text(
        "# Generated candidates: review before moving into the active adversarial suite.\n"
        + yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return output
