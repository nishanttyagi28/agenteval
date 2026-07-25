"""AST → QueryFacts, including scope-aware CTE traversal."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope

from agenteval.sql.parser import ParseResult, parse_sql

_WRITE = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Alter, exp.TruncateTable)


@dataclass
class ScopeFacts:
    """Facts for one SELECT scope (outer query, CTE body, or subquery)."""

    label: str  # "outer" | "CTE" | "subquery"
    has_where: bool
    has_limit: bool
    has_offset: bool
    has_group: bool
    has_agg: bool
    has_star: bool
    join_count: int
    table_names: list[str]
    column_names: list[str]
    filter_preds: list[str]
    select_sql: str


@dataclass
class QueryFacts:
    """Normalized structural facts for one input SQL string."""

    raw_sql: str
    dialect: str
    parsed: bool
    parse_error: str | None = None
    statement_count: int = 0
    roots: list[exp.Expression] = field(default_factory=list, repr=False)
    # Top-level write detection (any statement)
    write_kinds: list[str] = field(default_factory=list)
    # Disguised writes
    is_ctas: bool = False
    is_select_into: bool = False
    ctas_evidence: str = ""
    select_into_evidence: str = ""
    # Structural signals
    has_union: bool = False
    union_evidence: str = ""
    cartesian_evidences: list[str] = field(default_factory=list)
    windows_without_partition: list[str] = field(default_factory=list)
    scopes: list[ScopeFacts] = field(default_factory=list)
    has_comments: bool = False
    comment_evidence: str = ""
    # Aggregate views for diff-runs (outer-focused when outer scopes exist)
    tables: frozenset[str] = field(default_factory=frozenset)
    columns: frozenset[str] = field(default_factory=frozenset)
    filters: frozenset[str] = field(default_factory=frozenset)
    join_count: int = 0
    has_agg: bool = False


def table_qualified_name(table: exp.Table) -> str:
    """Schema-qualified name when available (sqlglot drops schema via ``.name`` alone)."""
    return f"{table.db}.{table.name}" if table.db else (table.name or table.sql())


def column_ref_name(col: exp.Column) -> str:
    """Prefer table.column when table is present."""
    name = col.name or col.sql()
    table = col.table
    if table:
        return f"{table}.{name}"
    return name


def _is_cartesian(join: exp.Join) -> bool:
    if join.args.get("on") or join.args.get("using"):
        return False
    return (join.args.get("method") or "").upper() != "NATURAL"


def _snip(node: exp.Expression, dialect: str, limit: int = 160) -> str:
    try:
        return node.sql(dialect=dialect)[:limit]
    except Exception:  # noqa: BLE001
        return type(node).__name__


def _scope_label(scope) -> str:
    if scope.is_cte:
        return "CTE"
    if scope.is_subquery:
        return "subquery"
    return "outer"


def _split_and_preds(where: exp.Expression | None, dialect: str) -> list[str]:
    """Flatten top-level AND predicates from a WHERE clause for set-diff."""
    if where is None:
        return []
    node = where.this if isinstance(where, exp.Where) else where
    preds: list[str] = []

    def walk(n: exp.Expression) -> None:
        if isinstance(n, exp.And):
            walk(n.left)
            walk(n.right)
        else:
            preds.append(_snip(n, dialect, limit=200))

    walk(node)
    return preds


def _columns_in_select(sel: exp.Select, dialect: str) -> list[str]:
    names: list[str] = []
    for proj in sel.expressions or []:
        if isinstance(proj, exp.Star) or isinstance(proj, exp.Column) and isinstance(proj.this, exp.Star):
            names.append("*")
            continue
        if isinstance(proj, exp.Alias):
            # Keep both alias and underlying column when simple.
            if proj.alias:
                names.append(str(proj.alias))
            inner = proj.this
            if isinstance(inner, exp.Column):
                names.append(column_ref_name(inner))
            continue
        if isinstance(proj, exp.Column):
            names.append(column_ref_name(proj))
            continue
        # Fallback: any column refs inside the projection.
        for col in proj.find_all(exp.Column):
            names.append(column_ref_name(col))
    # Also columns referenced in WHERE (filter side-channels).
    where = sel.args.get("where")
    if where is not None:
        for col in where.find_all(exp.Column):
            names.append(column_ref_name(col))
    # Stable unique order
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _extract_scopes(root: exp.Expression, dialect: str) -> list[ScopeFacts]:
    out: list[ScopeFacts] = []
    try:
        scopes = list(traverse_scope(root))
    except Exception:  # noqa: BLE001
        return out

    for scope in scopes:
        sel = scope.expression
        if not isinstance(sel, exp.Select):
            continue
        # scope.find_all is scope-local (does not leak CTE body into outer).
        names = [
            table_qualified_name(t)
            for t in scope.tables
            if isinstance(t, exp.Table) and t.name
        ]
        where = sel.args.get("where")
        out.append(
            ScopeFacts(
                label=_scope_label(scope),
                has_where=where is not None,
                has_limit=sel.args.get("limit") is not None,
                has_offset=sel.args.get("offset") is not None,
                has_group=sel.args.get("group") is not None,
                has_agg=any(True for _ in scope.find_all(exp.AggFunc)),
                has_star=any(True for _ in scope.find_all(exp.Star)),
                join_count=sum(1 for _ in scope.find_all(exp.Join)),
                table_names=names,
                column_names=_columns_in_select(sel, dialect),
                filter_preds=_split_and_preds(where, dialect),
                select_sql=_snip(sel, dialect),
            )
        )
    return out


def _analyze_root(root: exp.Expression, dialect: str, facts: QueryFacts) -> None:
    if isinstance(root, _WRITE):
        facts.write_kinds.append(type(root).__name__)
        return

    if isinstance(root, exp.Create) and root.expression is not None:
        facts.is_ctas = True
        facts.ctas_evidence = _snip(root, dialect)

    if isinstance(root, exp.Select) and root.args.get("into"):
        facts.is_select_into = True
        facts.select_into_evidence = _snip(root, dialect)

    for u in root.find_all(exp.Union):
        facts.has_union = True
        facts.union_evidence = _snip(u if u.parent is None else root, dialect)
        break

    for join in root.find_all(exp.Join):
        if _is_cartesian(join):
            parent = join.parent
            facts.cartesian_evidences.append(
                _snip(parent, dialect) if parent is not None else _snip(join, dialect)
            )

    for window in root.find_all(exp.Window):
        if not window.args.get("partition_by"):
            facts.windows_without_partition.append(_snip(window, dialect))

    facts.scopes.extend(_extract_scopes(root, dialect))


def _finalize_aggregates(facts: QueryFacts) -> None:
    """Fill tables/columns/filters/join_count from outer scopes (fallback: all)."""
    outer = [s for s in facts.scopes if s.label == "outer"]
    use = outer if outer else facts.scopes
    tables: set[str] = set()
    columns: set[str] = set()
    filters: set[str] = set()
    join_count = 0
    has_agg = False
    for s in use:
        tables.update(s.table_names)
        columns.update(s.column_names)
        filters.update(s.filter_preds)
        join_count += s.join_count
        has_agg = has_agg or s.has_agg
    facts.tables = frozenset(tables)
    facts.columns = frozenset(columns)
    facts.filters = frozenset(filters)
    facts.join_count = join_count
    facts.has_agg = has_agg


def extract_facts(sql: str, dialect: str = "postgres") -> QueryFacts:
    """Parse and normalize ``sql`` into :class:`QueryFacts`."""
    comment_m = re.search(r"--.*|/\*.*?\*/", sql, re.S)
    has_comments = bool(re.search(r"--|/\*", sql))

    result: ParseResult = parse_sql(sql, dialect=dialect)
    facts = QueryFacts(
        raw_sql=sql,
        dialect=dialect,
        parsed=result.ok,
        parse_error=result.error,
        statement_count=len(result.statements),
        roots=list(result.statements),
        has_comments=has_comments,
        comment_evidence=(comment_m.group(0) if comment_m else "")[:120],
    )
    if not result.ok:
        return facts

    for root in result.statements:
        _analyze_root(root, dialect, facts)
    _finalize_aggregates(facts)
    return facts
