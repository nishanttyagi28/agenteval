"""``agenteval compare-models`` — run the same golden suite across multiple
already-registered agents and lay their metrics side by side.

This is an orchestration layer only: each "model/provider" is an existing
``agents.yaml`` entry (already pointing at one model/provider via its own
adapter and ``adapter_options``, exactly like ``agenteval run``). Adapter
loading, suite execution, and scoring are all reused as-is from
``core.registry``, ``core.runner``, and ``core.store`` — no new provider
SDKs and no new scoring logic live here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agenteval.core.config import AgentDependencyNotFound
from agenteval.core.schema import AgentConfig, RunReport

_COLUMNS: tuple[tuple[str, str], ...] = (
    ("correctness_rate", "Correctness"),
    ("hallucination_rate", "Hallucination"),
    ("tool_call_accuracy", "Tool accuracy"),
    ("total_cost_usd", "Cost (USD)"),
    ("latency_p95_ms", "Latency p95 (ms)"),
)


@dataclass
class ModelComparisonRow:
    """One registered agent's outcome in a model/provider comparison."""

    agent: str
    display_name: str
    status: str  # "ok" | "error"
    error: str | None = None
    report: RunReport | None = None
    run_path: Path | None = None


def run_model_comparison(
    configs: list[AgentConfig],
    *,
    cases_path: str | Path,
    registry_path: str | Path,
    runs_dir_override: str | Path | None = None,
    agent_repo_overrides: dict[str, str | Path] | None = None,
    use_llm_judge: bool = True,
    quiet: bool = False,
) -> list[ModelComparisonRow]:
    """Run ``cases_path`` against every config in ``configs`` and collect rows.

    A configuration or execution error for one agent (bad adapter path,
    missing dependency, adapter construction failure) produces an error row
    instead of aborting the whole comparison, so one broken entry doesn't
    hide results for the others.
    """
    from agenteval.cli import _configured_path, _expand_adapter_options
    from agenteval.core.registry import load_adapter_class, resolve_agent_repository
    from agenteval.core.runner import run_golden_suite
    from agenteval.core.store import save_run_report

    cases_file = Path(cases_path)
    if not cases_file.is_file():
        raise ValueError(f"golden suite not found: {cases_file}")

    registry_file = Path(registry_path)
    overrides = agent_repo_overrides or {}
    rows: list[ModelComparisonRow] = []

    for config in configs:
        if not quiet:
            print(f"=== {config.name} ===")
        try:
            agent_repo = resolve_agent_repository(
                config,
                explicit=overrides.get(config.name),
                registry_path=registry_file,
            )
            options = _expand_adapter_options(config.adapter_options, agent_repo)
            adapter = load_adapter_class(config.adapter)(repo_path=agent_repo, **options)
            report = run_golden_suite(
                adapter,
                cases_path=cases_file,
                adapter_name=config.name,
                verbose=not quiet,
                use_llm_judge=use_llm_judge,
            )
            runs_dir = (
                Path(runs_dir_override)
                if runs_dir_override
                else _configured_path(registry_file, config.runs_dir)
            )
            # save_run_report's default filename is <timestamp>_<git_sha>.json,
            # which collides when multiple agents share one runs_dir (as
            # compare-models allows) and finish within the same second —
            # report.run_id also carries a per-run uuid suffix, so it stays
            # unique even then.
            run_path = save_run_report(report, runs_dir=runs_dir, filename=f"{report.run_id}.json")
        except AgentDependencyNotFound as exc:
            rows.append(
                ModelComparisonRow(
                    agent=config.name, display_name=config.display_name, status="error", error=str(exc)
                )
            )
            continue
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            rows.append(
                ModelComparisonRow(
                    agent=config.name,
                    display_name=config.display_name,
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        rows.append(
            ModelComparisonRow(
                agent=config.name,
                display_name=config.display_name,
                status="ok",
                report=report,
                run_path=run_path,
            )
        )
    return rows


def _fmt(value: Any, key: str) -> str:
    if value is None:
        return "n/a"
    if key in {"correctness_rate", "hallucination_rate", "tool_call_accuracy"}:
        return f"{value * 100:.1f}%"
    if key == "total_cost_usd":
        return f"${value:.6f}"
    if key == "latency_p95_ms":
        return f"{value:.0f}"
    return str(value)


def format_comparison_table(rows: list[ModelComparisonRow]) -> str:
    """Render a Markdown table: one row per agent, one column per metric."""
    header = ["Agent", "Status", *(label for _, label in _COLUMNS)]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    errors: list[str] = []
    for row in rows:
        if row.status == "ok" and row.report is not None:
            cells = [_fmt(getattr(row.report, key), key) for key, _ in _COLUMNS]
        else:
            cells = ["-" for _ in _COLUMNS]
            errors.append(f"- `{row.agent}`: {row.error}")
        lines.append("| " + " | ".join([row.agent, row.status, *cells]) + " |")
    if errors:
        lines.append("")
        lines.append("Errors:")
        lines.extend(errors)
    return "\n".join(lines) + "\n"


def comparison_to_dict(rows: list[ModelComparisonRow]) -> dict[str, Any]:
    """Machine-readable comparison payload for ``--json-out``."""
    agents: list[dict[str, Any]] = []
    for row in rows:
        entry: dict[str, Any] = {
            "agent": row.agent,
            "display_name": row.display_name,
            "status": row.status,
            "error": row.error,
            "run_path": str(row.run_path) if row.run_path else None,
        }
        for key, _ in _COLUMNS:
            entry[key] = getattr(row.report, key) if row.report is not None else None
        agents.append(entry)
    return {"agents": agents}


def write_outputs(
    rows: list[ModelComparisonRow],
    *,
    json_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
) -> None:
    if json_path is not None:
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(comparison_to_dict(rows), indent=2) + "\n", encoding="utf-8")
    if markdown_path is not None:
        path = Path(markdown_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(format_comparison_table(rows), encoding="utf-8")


__all__ = [
    "ModelComparisonRow",
    "run_model_comparison",
    "format_comparison_table",
    "comparison_to_dict",
    "write_outputs",
]
