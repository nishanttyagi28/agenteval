"""``agenteval import`` — convert an external CSV dataset into golden test cases.

A small "mapping config" YAML declares which CSV columns become which parts
of a golden :class:`~agenteval.core.schema.TestCase` (prompt, ground truth,
correctness type, required tools, ...). This module only reads a CSV and a
mapping file and writes a golden YAML suite; it never makes a live provider
call. Rows are read as plain strings (``dtype=str``) so numeric ground truth
still round-trips through the existing numeric-correctness machinery in
``core.metrics`` without pandas' own float/NaN coercion getting involved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agenteval.core._fsutil import atomic_write_text
from agenteval.core.generator import test_case_to_dict
from agenteval.core.schema import CorrectnessType, Expects, TestCase


class DatasetImportError(ValueError):
    """Raised for invalid mapping configs, CSV rows, or CLI input."""


@dataclass(frozen=True)
class ImportMapping:
    """Column-to-golden-case-field mapping, loaded from a small YAML config."""

    prompt_column: str
    ground_truth_column: str
    id_column: str | None = None
    correctness_type: str = "exact"
    numeric_tolerance: float = 0.01
    tags: tuple[str, ...] = field(default_factory=tuple)
    must_call_tools_column: str | None = None
    must_not_hallucinate: bool = False

    @classmethod
    def from_dict(cls, data: Any) -> ImportMapping:
        if not isinstance(data, dict):
            raise DatasetImportError("mapping config must be a YAML mapping")

        prompt_column = data.get("prompt_column")
        if not isinstance(prompt_column, str) or not prompt_column.strip():
            raise DatasetImportError("mapping config: prompt_column is required")
        ground_truth_column = data.get("ground_truth_column")
        if not isinstance(ground_truth_column, str) or not ground_truth_column.strip():
            raise DatasetImportError("mapping config: ground_truth_column is required")

        id_column = data.get("id_column")
        if id_column is not None and (not isinstance(id_column, str) or not id_column.strip()):
            raise DatasetImportError("mapping config: id_column must be a non-empty string or omitted")

        correctness_type = str(data.get("correctness_type", "exact"))
        try:
            CorrectnessType(correctness_type)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in CorrectnessType)
            raise DatasetImportError(
                f"mapping config: correctness_type must be one of {allowed}, got {correctness_type!r}"
            ) from exc
        if correctness_type == CorrectnessType.numeric_table.value:
            raise DatasetImportError(
                "mapping config: correctness_type 'numeric_table' is not supported by CSV "
                "import — a single flat cell cannot hold a table-shaped ground truth"
            )

        try:
            numeric_tolerance = float(data.get("numeric_tolerance", 0.01))
        except (TypeError, ValueError) as exc:
            raise DatasetImportError("mapping config: numeric_tolerance must be a number") from exc

        tags_raw = data.get("tags") or []
        if not isinstance(tags_raw, list) or not all(isinstance(item, str) for item in tags_raw):
            raise DatasetImportError("mapping config: tags must be a list of strings")

        must_call_tools_column = data.get("must_call_tools_column")
        if must_call_tools_column is not None and (
            not isinstance(must_call_tools_column, str) or not must_call_tools_column.strip()
        ):
            raise DatasetImportError(
                "mapping config: must_call_tools_column must be a non-empty string or omitted"
            )

        return cls(
            prompt_column=prompt_column.strip(),
            ground_truth_column=ground_truth_column.strip(),
            id_column=id_column.strip() if id_column else None,
            correctness_type=correctness_type,
            numeric_tolerance=numeric_tolerance,
            tags=tuple(tags_raw),
            must_call_tools_column=must_call_tools_column.strip() if must_call_tools_column else None,
            must_not_hallucinate=bool(data.get("must_not_hallucinate", False)),
        )


def load_mapping(path: str | Path) -> ImportMapping:
    """Load and validate a mapping config YAML file."""
    import yaml

    mapping_path = Path(path)
    try:
        raw = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetImportError(f"mapping config not found: {mapping_path}") from exc
    except yaml.YAMLError as exc:
        raise DatasetImportError(f"invalid mapping config YAML in {mapping_path}: {exc}") from exc
    return ImportMapping.from_dict(raw)


_MAPPING_TEMPLATE = """\
# agenteval import mapping config.
# Map CSV columns onto golden test-case fields, then run:
#   agenteval import <file.csv> --mapping <this file> --output tests/golden/imported.yaml

prompt_column: question          # required: CSV column holding the prompt
ground_truth_column: answer      # required: CSV column holding the expected answer
id_column: id                    # optional: CSV column for stable case ids (auto-generated if omitted)
correctness_type: exact          # exact | numeric | contains | numeric_table | llm_judge
numeric_tolerance: 0.01          # only used when correctness_type: numeric
tags: [imported]                 # static tags applied to every imported case
must_call_tools_column: null     # optional: CSV column with comma-separated required tool names
must_not_hallucinate: false
"""


def emit_mapping_template(path: str | Path, *, force: bool = False) -> Path:
    """Write a starter, commented mapping config for the user to edit."""
    out_path = Path(path)
    if out_path.exists() and not force:
        raise DatasetImportError(f"{out_path} already exists (use --overwrite)")
    return atomic_write_text(out_path, _MAPPING_TEMPLATE)


def _read_csv_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    import pandas as pd

    try:
        frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_filter=False)
    except FileNotFoundError as exc:
        raise DatasetImportError(f"CSV file not found: {csv_path}") from exc
    except pd.errors.EmptyDataError as exc:
        raise DatasetImportError(f"CSV file is empty: {csv_path}") from exc
    columns = [str(column) for column in frame.columns]
    rows = frame.to_dict(orient="records")
    return columns, rows


def import_csv(csv_path: str | Path, mapping: ImportMapping) -> list[TestCase]:
    """Read ``csv_path`` and produce golden ``TestCase`` objects per ``mapping``.

    Every row must yield a non-empty prompt and ground truth: a CSV import is
    meant to become real golden data (not a best-effort proposal), so a bad
    row fails the whole import loudly and specifically rather than silently
    dropping a case or producing an incomplete suite.
    """
    path = Path(csv_path)
    columns, rows = _read_csv_rows(path)

    required_columns = [mapping.prompt_column, mapping.ground_truth_column]
    if mapping.id_column:
        required_columns.append(mapping.id_column)
    if mapping.must_call_tools_column:
        required_columns.append(mapping.must_call_tools_column)
    missing = [column for column in required_columns if column not in columns]
    if missing:
        raise DatasetImportError(
            f"CSV is missing configured column(s) {missing}; available columns: {columns}"
        )
    if not rows:
        raise DatasetImportError(f"CSV file has no data rows: {path}")

    cases: list[TestCase] = []
    seen_ids: dict[str, int] = {}
    for index, row in enumerate(rows, start=1):
        prompt = str(row.get(mapping.prompt_column, "")).strip()
        if not prompt:
            raise DatasetImportError(f"row {index}: {mapping.prompt_column!r} is empty")
        ground_truth = str(row.get(mapping.ground_truth_column, "")).strip()
        if not ground_truth:
            raise DatasetImportError(f"row {index}: {mapping.ground_truth_column!r} is empty")

        if mapping.id_column:
            case_id = str(row.get(mapping.id_column, "")).strip()
            if not case_id:
                raise DatasetImportError(f"row {index}: {mapping.id_column!r} is empty")
        else:
            case_id = f"row_{index}"
        if case_id in seen_ids:
            raise DatasetImportError(
                f"duplicate case id {case_id!r} (rows {seen_ids[case_id]} and {index})"
            )
        seen_ids[case_id] = index

        must_call_tools: list[str] = []
        if mapping.must_call_tools_column:
            raw_tools = str(row.get(mapping.must_call_tools_column, ""))
            must_call_tools = [tool.strip() for tool in raw_tools.split(",") if tool.strip()]

        if mapping.correctness_type == CorrectnessType.numeric.value:
            from agenteval.core.metrics import extract_numbers

            if not extract_numbers(ground_truth):
                raise DatasetImportError(
                    f"row {index}: ground truth {ground_truth!r} is not numeric "
                    f"(correctness_type: numeric)"
                )

        expects = Expects.from_dict(
            {
                "correctness_type": mapping.correctness_type,
                "ground_truth": ground_truth,
                "numeric_tolerance": mapping.numeric_tolerance,
                "must_call_tools": must_call_tools,
                "must_not_hallucinate": mapping.must_not_hallucinate,
            }
        )
        cases.append(
            TestCase(
                id=case_id,
                prompt=prompt,
                expects=expects,
                tags=list(mapping.tags),
            )
        )
    return cases


def write_golden_yaml(cases: list[TestCase], path: str | Path, *, force: bool = False) -> Path:
    """Write imported cases as a golden YAML suite, loadable by ``schema.load_test_cases``."""
    import yaml

    if not cases:
        raise DatasetImportError("no cases to write")
    out_path = Path(path)
    if out_path.exists() and not force:
        raise DatasetImportError(f"{out_path} already exists (use --overwrite)")
    payload = [test_case_to_dict(case) for case in cases]
    header = (
        "# Imported from an external dataset via `agenteval import`.\n"
        "# Review before relying on it as a blocking golden suite.\n"
    )
    content = header + yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    return atomic_write_text(out_path, content)


__all__ = [
    "DatasetImportError",
    "ImportMapping",
    "load_mapping",
    "emit_mapping_template",
    "import_csv",
    "write_golden_yaml",
]
