"""SQL policy file loader (blueprint Section 7) — Tier 2+ configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class PolicyError(ValueError):
    """Raised for malformed or incomplete policy files (never crash uncaught)."""


@dataclass
class Policy:
    """Parsed sql-policy.yml."""

    path: Path
    version: int = 1
    dialect: str | None = None
    schema_file: Path | None = None
    # statements
    allow_statements: set[str] = field(default_factory=lambda: {"select"})
    block_statements: set[str] = field(default_factory=set)
    # schemas / tables
    block_schemas: set[str] = field(default_factory=set)
    block_tables: set[str] = field(default_factory=set)
    # columns → category, categories → severity
    columns: dict[str, str] = field(default_factory=dict)  # "schema.table.col" or "table.col" → cat
    categories: dict[str, str] = field(default_factory=dict)  # cat → "block"|"review"
    # per-rule severity overrides (SQL401-403 cannot be block — enforced elsewhere)
    rule_overrides: dict[str, str] = field(default_factory=dict)
    # Tier 3
    sandbox_dsn: str | None = None
    explain_cost_budget: float = 1_000_000.0
    explain_timeout_ms: int = 5_000
    # Tier 4
    session_max_queries: int = 50
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class SchemaCatalog:
    """Table → set of column names (lowercased for matching)."""

    # key: "schema.table" or "table" (lowercase)
    tables: dict[str, set[str]] = field(default_factory=dict)
    path: Path | None = None

    def has_table(self, name: str) -> bool:
        return _norm_ident(name) in self.tables or name.lower() in self.tables

    def columns_for(self, table: str) -> set[str] | None:
        key = _norm_ident(table)
        if key in self.tables:
            return self.tables[key]
        # bare table match against *.table
        bare = key.split(".")[-1]
        for t, cols in self.tables.items():
            if t == bare or t.endswith("." + bare):
                return cols
        return None

    def to_sqlglot_schema(self) -> dict[str, dict[str, str]]:
        """Map to sqlglot qualify() schema: {table: {col: type}}."""
        out: dict[str, dict[str, str]] = {}
        for table, cols in self.tables.items():
            # sqlglot often wants bare table names for simple schemas
            bare = table.split(".")[-1]
            entry = out.setdefault(bare, {})
            for c in cols:
                entry[c] = "TEXT"
            # also register schema.table form if present
            if "." in table:
                out.setdefault(table, {c: "TEXT" for c in cols})
        return out


def _norm_ident(name: str) -> str:
    return name.strip().strip('"').strip("'").lower()


def _as_str_list(val: Any, field_name: str) -> list[str]:
    if val is None:
        return []
    if isinstance(val, str):
        return [val]
    if isinstance(val, list):
        out = []
        for i, item in enumerate(val):
            if not isinstance(item, str):
                raise PolicyError(f"{field_name}[{i}] must be a string, got {type(item).__name__}")
            out.append(item)
        return out
    raise PolicyError(f"{field_name} must be a list of strings or a string, got {type(val).__name__}")


def _as_str_dict(val: Any, field_name: str) -> dict[str, str]:
    if val is None:
        return {}
    if not isinstance(val, dict):
        raise PolicyError(f"{field_name} must be a mapping, got {type(val).__name__}")
    out: dict[str, str] = {}
    for k, v in val.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise PolicyError(f"{field_name} keys and values must be strings")
        out[k] = v
    return out


def load_schema_file(path: Path | str) -> SchemaCatalog:
    """Load a schema catalog from YAML/JSON.

    Expected shape::

        tables:
          public.users:
            - id
            - email
          orders:
            columns: [id, user_id, total]
    """
    p = Path(path)
    if not p.is_file():
        raise PolicyError(f"schema_file not found: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8-sig"))
    except yaml.YAMLError as exc:
        raise PolicyError(f"schema_file is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise PolicyError("schema_file root must be a mapping")
    tables_raw = data.get("tables")
    if tables_raw is None:
        raise PolicyError("schema_file missing required key 'tables'")
    if not isinstance(tables_raw, dict):
        raise PolicyError("schema_file 'tables' must be a mapping")

    tables: dict[str, set[str]] = {}
    for tname, cols in tables_raw.items():
        if not isinstance(tname, str):
            raise PolicyError("schema table names must be strings")
        col_list: list[str]
        if isinstance(cols, dict) and "columns" in cols:
            col_list = _as_str_list(cols["columns"], f"tables.{tname}.columns")
        elif isinstance(cols, list):
            col_list = _as_str_list(cols, f"tables.{tname}")
        else:
            raise PolicyError(
                f"tables.{tname} must be a list of columns or {{columns: [...]}}"
            )
        tables[_norm_ident(tname)] = {_norm_ident(c) for c in col_list}
    return SchemaCatalog(tables=tables, path=p)


def load_policy(path: Path | str) -> Policy:
    """Parse and validate sql-policy.yml. Raises :class:`PolicyError` on bad input."""
    p = Path(path)
    if not p.is_file():
        raise PolicyError(f"policy file not found: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8-sig"))
    except yaml.YAMLError as exc:
        raise PolicyError(f"policy file is not valid YAML: {exc}") from exc
    if data is None:
        raise PolicyError("policy file is empty")
    if not isinstance(data, dict):
        raise PolicyError("policy file root must be a mapping")

    version = data.get("version", 1)
    if not isinstance(version, int):
        raise PolicyError("version must be an integer")

    schema_file: Path | None = None
    if data.get("schema_file"):
        if not isinstance(data["schema_file"], str):
            raise PolicyError("schema_file must be a string path")
        candidate = Path(data["schema_file"])
        if not candidate.is_file():
            # resolve relative to policy file directory
            candidate = (p.parent / data["schema_file"]).resolve()
        schema_file = candidate

    statements = data.get("statements") or data.get("allow") or {}
    # support both nested and flat shapes
    if isinstance(statements, dict) and (
        "allow" in statements or "block" in statements
    ):
        allow = _as_str_list(statements.get("allow"), "statements.allow")
        block = _as_str_list(statements.get("block"), "statements.block")
    else:
        # top-level allow.statements from docs sample
        allow_block = data.get("allow") or {}
        deny_block = data.get("deny") or {}
        if isinstance(allow_block, dict) and "statements" in allow_block:
            allow = _as_str_list(allow_block.get("statements"), "allow.statements")
        else:
            allow = ["select"]
        block = []
        if isinstance(deny_block, dict) and "statements" in deny_block:
            block = _as_str_list(deny_block.get("statements"), "deny.statements")

    # schemas
    block_schemas: list[str] = []
    schemas = data.get("schemas") or {}
    if isinstance(schemas, dict):
        block_schemas = _as_str_list(schemas.get("block"), "schemas.block")
    deny = data.get("deny") or {}
    if isinstance(deny, dict) and "schemas" in deny:
        block_schemas.extend(_as_str_list(deny.get("schemas"), "deny.schemas"))

    # tables
    block_tables: list[str] = []
    tables = data.get("tables") or {}
    if isinstance(tables, dict):
        block_tables = _as_str_list(tables.get("block"), "tables.block")
    if isinstance(deny, dict) and "tables" in deny:
        block_tables.extend(_as_str_list(deny.get("tables"), "deny.tables"))

    # columns → category
    columns: dict[str, str] = {}
    cols_raw = data.get("columns") or {}
    if isinstance(cols_raw, dict):
        # map form: table.col: category OR category lists
        for k, v in cols_raw.items():
            if isinstance(v, str):
                columns[_norm_ident(str(k))] = v.lower()
            elif isinstance(v, list):
                # category: [col, col]
                for item in v:
                    if isinstance(item, str):
                        columns[_norm_ident(item)] = str(k).lower()
            else:
                raise PolicyError(f"columns.{k} has unsupported value type")
    if isinstance(deny, dict) and "columns" in deny:
        for item in _as_str_list(deny.get("columns"), "deny.columns"):
            # treat deny.columns as restricted
            columns[_norm_ident(item)] = "restricted"

    categories = {
        _norm_ident(k): v.lower()
        for k, v in _as_str_dict(data.get("categories"), "categories").items()
    }
    if not categories:
        categories = {"restricted": "block", "sensitive": "review"}

    for cat, sev in categories.items():
        if sev not in ("block", "review"):
            raise PolicyError(
                f"categories.{cat} severity must be 'block' or 'review', got {sev!r}"
            )

    rule_overrides = {
        k.upper(): v.lower()
        for k, v in _as_str_dict(data.get("rules"), "rules").items()
    }
    for rid, sev in rule_overrides.items():
        if sev not in ("block", "review"):
            raise PolicyError(f"rules.{rid} must be 'block' or 'review', got {sev!r}")
        if rid in ("SQL401", "SQL402", "SQL403") and sev == "block":
            raise PolicyError(
                f"rules.{rid} cannot be set to 'block' — Tier 5 is advisory-only"
            )

    execution = data.get("execution") or {}
    if execution is None:
        execution = {}
    if not isinstance(execution, dict):
        raise PolicyError("execution must be a mapping")
    sandbox_dsn = execution.get("sandbox_dsn") or data.get("sandbox_dsn")
    if sandbox_dsn is not None and not isinstance(sandbox_dsn, str):
        raise PolicyError("execution.sandbox_dsn must be a string")
    cost_budget = execution.get("cost_budget", execution.get("explain_cost_budget", 1_000_000))
    try:
        cost_budget_f = float(cost_budget)
    except (TypeError, ValueError) as exc:
        raise PolicyError("execution.cost_budget must be a number") from exc
    timeout_ms = execution.get("timeout_ms", 5_000)
    try:
        timeout_ms_i = int(timeout_ms)
    except (TypeError, ValueError) as exc:
        raise PolicyError("execution.timeout_ms must be an integer") from exc

    session = data.get("session") or {}
    if not isinstance(session, dict):
        raise PolicyError("session must be a mapping")
    max_q = session.get("max_queries", data.get("session_max_queries", 50))
    try:
        max_q_i = int(max_q)
    except (TypeError, ValueError) as exc:
        raise PolicyError("session.max_queries must be an integer") from exc

    dialect = data.get("dialect")
    if dialect is not None and not isinstance(dialect, str):
        raise PolicyError("dialect must be a string")

    return Policy(
        path=p,
        version=version,
        dialect=dialect,
        schema_file=schema_file,
        allow_statements={s.lower() for s in allow} if allow else {"select"},
        block_statements={s.lower() for s in block},
        block_schemas={_norm_ident(s) for s in block_schemas},
        block_tables={_norm_ident(t) for t in block_tables},
        columns=columns,
        categories=categories,
        rule_overrides=rule_overrides,
        sandbox_dsn=sandbox_dsn,
        explain_cost_budget=cost_budget_f,
        explain_timeout_ms=timeout_ms_i,
        session_max_queries=max_q_i,
        raw=data,
    )


def severity_for_category(policy: Policy, category: str) -> str:
    """Return block/review for a category (default review)."""
    return policy.categories.get(_norm_ident(category), "review")


def column_category(policy: Policy, qualified_col: str) -> str | None:
    """Look up category for a column; supports table.col and *.col wildcards."""
    key = _norm_ident(qualified_col)
    if key in policy.columns:
        return policy.columns[key]
    # bare column
    bare = key.split(".")[-1]
    if bare in policy.columns:
        return policy.columns[bare]
    # wildcard *.col
    wild = f"*.{bare}"
    if wild in policy.columns:
        return policy.columns[wild]
    # suffix match table.col against schema.table.col keys
    for pk, cat in policy.columns.items():
        if pk == key or pk.endswith("." + key) or key.endswith("." + pk):
            return cat
        if pk.endswith("." + bare) or pk == bare:
            return cat
    return None
