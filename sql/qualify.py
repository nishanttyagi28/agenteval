"""Schema-aware column qualification via sqlglot.optimizer.qualify (per-scope)."""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlglot import exp
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import traverse_scope

from agenteval.sql.normalize import table_qualified_name
from agenteval.sql.parser import parse_sql
from agenteval.sql.policy import SchemaCatalog


@dataclass
class ResolvedColumn:
    """A column reference resolved to a real table when possible."""

    raw: str  # as written / after qualify (may use alias)
    column: str  # bare column name (unquoted, lower for matching)
    table_alias: str | None
    real_table: str | None  # schema.table or table
    qualified: str | None  # real_table.column when known
    quoted: bool
    original_case: str  # column name as it appeared (for SQL106)
    resolved: bool


@dataclass
class QualifyResult:
    ok: bool
    dialect: str
    columns: list[ResolvedColumn] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    alias_map: dict[str, str] = field(default_factory=dict)  # alias → real table
    error: str | None = None
    unresolved_count: int = 0


def _strip_quotes(name: str) -> str:
    return name.strip().strip('"').strip("'").strip("`")


def _is_quoted_ident(node: exp.Identifier | None) -> bool:
    if node is None:
        return False
    return bool(getattr(node, "quoted", False) or node.args.get("quoted"))


def build_alias_map(expression: exp.Expression) -> dict[str, str]:
    """Map table aliases → schema-qualified real table names (scope-aware)."""
    mapping: dict[str, str] = {}
    try:
        scopes = list(traverse_scope(expression))
    except Exception:  # noqa: BLE001
        scopes = []
    for scope in scopes:
        sources = getattr(scope, "sources", {}) or {}
        for alias, source in sources.items():
            alias_key = _strip_quotes(str(alias)).lower()
            if isinstance(source, exp.Table):
                mapping[alias_key] = table_qualified_name(source)
            elif isinstance(source, exp.Expression) and isinstance(
                getattr(source, "this", None), exp.Table
            ):
                mapping[alias_key] = table_qualified_name(source.this)
    # Also walk FROM/JOIN tables for unaliased names
    for table in expression.find_all(exp.Table):
        if not table.name:
            continue
        real = table_qualified_name(table)
        mapping[_strip_quotes(table.name).lower()] = real
        if table.alias:
            mapping[_strip_quotes(str(table.alias)).lower()] = real
    return mapping


def resolve_columns(
    expression: exp.Expression,
    alias_map: dict[str, str],
) -> list[ResolvedColumn]:
    """Resolve Column nodes using alias→table map (not blindly root.find)."""
    out: list[ResolvedColumn] = []
    try:
        scopes = list(traverse_scope(expression))
    except Exception:  # noqa: BLE001
        scopes = []

    # If only one physical table is in play, use it for bare columns.
    unique_tables = list(dict.fromkeys(alias_map.values()))
    sole_table = unique_tables[0] if len(unique_tables) == 1 else None

    seen: set[tuple[str, str | None, str]] = set()

    def handle_column(col: exp.Column) -> None:
        col_ident = col.this if isinstance(col.this, exp.Identifier) else None
        raw_name = col.name or (col_ident.name if col_ident else col.sql())
        original_case = _strip_quotes(str(raw_name))
        quoted = _is_quoted_ident(col_ident)
        # Also treat "EMAIL" form in sql() as quoted
        if not quoted and col.sql().startswith('"'):
            quoted = True
        table_alias = col.table
        if table_alias:
            table_alias = _strip_quotes(str(table_alias))
        real_table = None
        if table_alias:
            real_table = alias_map.get(table_alias.lower()) or table_alias
        elif sole_table is not None:
            real_table = sole_table
        bare = original_case.lower()
        qualified = f"{real_table}.{bare}" if real_table else None
        key = (bare, table_alias.lower() if table_alias else None, qualified or "")
        if key in seen:
            return
        seen.add(key)
        out.append(
            ResolvedColumn(
                raw=col.sql(),
                column=bare,
                table_alias=table_alias,
                real_table=real_table,
                qualified=qualified.lower() if qualified else None,
                quoted=quoted,
                original_case=original_case,
                resolved=bool(real_table),
            )
        )

    if scopes:
        for scope in scopes:
            # Per-scope sole table for CTE isolation
            scope_tables = []
            for alias, source in (getattr(scope, "sources", {}) or {}).items():
                if isinstance(source, exp.Table):
                    scope_tables.append(table_qualified_name(source))
            scope_sole = scope_tables[0] if len(scope_tables) == 1 else sole_table
            for col in scope.find_all(exp.Column):
                # temporarily prefer scope_sole
                nonlocal_sole = scope_sole
                col_ident = col.this if isinstance(col.this, exp.Identifier) else None
                raw_name = col.name or (col_ident.name if col_ident else col.sql())
                original_case = _strip_quotes(str(raw_name))
                quoted = _is_quoted_ident(col_ident) or col.sql().startswith('"')
                table_alias = col.table
                if table_alias:
                    table_alias = _strip_quotes(str(table_alias))
                real_table = None
                if table_alias:
                    real_table = alias_map.get(table_alias.lower()) or table_alias
                elif nonlocal_sole is not None:
                    real_table = nonlocal_sole
                bare = original_case.lower()
                qualified = f"{real_table}.{bare}" if real_table else None
                key = (bare, table_alias.lower() if table_alias else None, qualified or "")
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    ResolvedColumn(
                        raw=col.sql(),
                        column=bare,
                        table_alias=table_alias,
                        real_table=real_table,
                        qualified=qualified.lower() if qualified else None,
                        quoted=quoted,
                        original_case=original_case,
                        resolved=bool(real_table),
                    )
                )
    else:
        for col in expression.find_all(exp.Column):
            handle_column(col)
    return out


def qualify_sql(
    sql: str,
    *,
    dialect: str = "postgres",
    schema: SchemaCatalog | None = None,
) -> QualifyResult:
    """Parse + qualify SQL; build alias map; resolve columns per scope.

    Qualifies on the parsed AST (sqlglot qualify walks scopes). CTE bodies are
    handled via traverse_scope for alias mapping so outer/CTE do not leak.
    """
    parsed = parse_sql(sql, dialect=dialect)
    if not parsed.ok or not parsed.statements:
        return QualifyResult(
            ok=False,
            dialect=dialect,
            error=parsed.error or "parse failed",
        )

    all_cols: list[ResolvedColumn] = []
    all_tables: list[str] = []
    alias_map: dict[str, str] = {}
    sg_schema = schema.to_sqlglot_schema() if schema else None

    for root in parsed.statements:
        try:
            if sg_schema is not None:
                qualified = qualify(
                    root.copy(),
                    schema=sg_schema,
                    dialect=dialect,
                    identify=True,
                )
            else:
                # Still expand stars / qualify aliases when possible without schema
                try:
                    qualified = qualify(root.copy(), dialect=dialect, identify=False)
                except Exception:  # noqa: BLE001
                    qualified = root
        except Exception as exc:  # noqa: BLE001
            # Fall back to unqualified AST for partial resolution
            qualified = root
            _ = exc

        amap = build_alias_map(qualified)
        alias_map.update(amap)
        cols = resolve_columns(qualified, amap)
        all_cols.extend(cols)
        for t in amap.values():
            if t not in all_tables:
                all_tables.append(t)

    unresolved = sum(1 for c in all_cols if not c.resolved)
    return QualifyResult(
        ok=True,
        dialect=dialect,
        columns=all_cols,
        tables=all_tables,
        alias_map=alias_map,
        unresolved_count=unresolved,
    )
