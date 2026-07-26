"""SQLite schema versioning, upgrades, backup, and rollback helpers."""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

# DB schema version 3 = product V2.1 (occurrences, replay, minimization).
# Version 2 remains candidate lineage from V2 Failure Memory.
# Product AgentEval V2.1 uses database schema v3+ (v4 unique fingerprint, v5 delivery keys).
CURRENT_SCHEMA_VERSION = 5


@dataclass(frozen=True)
class Migration:
    version: int
    sql: str
    description: str

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.sql.encode("utf-8")).hexdigest()


# Version 1 — initial Failure Memory schema.
_V1_SQL = """
CREATE TABLE IF NOT EXISTS fm_schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL,
    checksum TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fm_traces (
    internal_id INTEGER PRIMARY KEY AUTOINCREMENT,
    external_trace_id TEXT NOT NULL UNIQUE,
    schema_version INTEGER NOT NULL,
    occurred_at TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    source TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    status TEXT NOT NULL,
    content_captured INTEGER NOT NULL DEFAULT 0,
    prompt TEXT,
    output TEXT,
    error_type TEXT,
    error_message TEXT,
    tool_calls_json TEXT NOT NULL DEFAULT '[]',
    metrics_json TEXT NOT NULL DEFAULT '{}',
    attributes_json TEXT NOT NULL DEFAULT '{}',
    failure_category TEXT,
    fingerprint TEXT,
    redaction_version INTEGER
);

CREATE INDEX IF NOT EXISTS idx_fm_traces_occurred_at ON fm_traces(occurred_at);
CREATE INDEX IF NOT EXISTS idx_fm_traces_agent_name ON fm_traces(agent_name);
CREATE INDEX IF NOT EXISTS idx_fm_traces_status ON fm_traces(status);
CREATE INDEX IF NOT EXISTS idx_fm_traces_failure_category ON fm_traces(failure_category);
CREATE INDEX IF NOT EXISTS idx_fm_traces_fingerprint ON fm_traces(fingerprint);

CREATE TABLE IF NOT EXISTS fm_spans (
    internal_id INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_internal_id INTEGER NOT NULL REFERENCES fm_traces(internal_id) ON DELETE CASCADE,
    sequence_number INTEGER NOT NULL,
    external_span_id TEXT,
    parent_span_id TEXT,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    status TEXT,
    input_json TEXT,
    output_json TEXT,
    duration_ms REAL,
    attributes_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE (trace_internal_id, sequence_number),
    UNIQUE (trace_internal_id, external_span_id)
);

CREATE TABLE IF NOT EXISTS fm_clusters (
    internal_id INTEGER PRIMARY KEY AUTOINCREMENT,
    stable_cluster_key TEXT NOT NULL UNIQUE,
    failure_category TEXT NOT NULL,
    title TEXT NOT NULL,
    representative_trace_id TEXT NOT NULL,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    occurrence_count INTEGER NOT NULL DEFAULT 0,
    review_state TEXT NOT NULL DEFAULT 'open',
    algorithm_version TEXT NOT NULL,
    explanation_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS fm_cluster_members (
    cluster_id INTEGER NOT NULL REFERENCES fm_clusters(internal_id) ON DELETE CASCADE,
    trace_id INTEGER NOT NULL REFERENCES fm_traces(internal_id) ON DELETE CASCADE,
    similarity_score REAL NOT NULL,
    assigned_at TEXT NOT NULL,
    algorithm_version TEXT NOT NULL,
    PRIMARY KEY (trace_id, algorithm_version),
    UNIQUE (cluster_id, trace_id)
);

CREATE TABLE IF NOT EXISTS fm_candidates (
    candidate_id TEXT PRIMARY KEY,
    cluster_id INTEGER NOT NULL REFERENCES fm_clusters(internal_id),
    representative_trace_id TEXT NOT NULL,
    state TEXT NOT NULL,
    stable_case_id TEXT,
    expected_behaviour_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    approved_at TEXT,
    rejected_at TEXT,
    exported_at TEXT,
    revision INTEGER NOT NULL DEFAULT 1
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_fm_candidates_active_cluster
    ON fm_candidates(cluster_id)
    WHERE state IN ('pending_review', 'approved', 'exported');

CREATE TABLE IF NOT EXISTS fm_review_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL REFERENCES fm_candidates(candidate_id),
    action TEXT NOT NULL,
    actor TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    note TEXT,
    previous_state TEXT,
    new_state TEXT NOT NULL,
    payload_checksum TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fm_exports (
    export_id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL REFERENCES fm_candidates(candidate_id),
    case_id TEXT NOT NULL,
    suite_path TEXT NOT NULL,
    case_checksum TEXT NOT NULL,
    exported_at TEXT NOT NULL,
    UNIQUE (candidate_id, case_id, suite_path)
);
"""

# Version 2 — candidate lineage for immutable approved revisions.
# Replaces the single-active-cluster unique index so an approved row can
# coexist with a new pending revision of the same cluster.
_V2_SQL = """
ALTER TABLE fm_candidates ADD COLUMN parent_candidate_id TEXT;
ALTER TABLE fm_candidates ADD COLUMN revision_of TEXT;
ALTER TABLE fm_candidates ADD COLUMN revision_idempotency_key TEXT;

DROP INDEX IF EXISTS idx_fm_candidates_active_cluster;

CREATE UNIQUE INDEX IF NOT EXISTS idx_fm_candidates_pending_cluster
    ON fm_candidates(cluster_id)
    WHERE state = 'pending_review';

CREATE UNIQUE INDEX IF NOT EXISTS idx_fm_candidates_revision_key
    ON fm_candidates(revision_idempotency_key)
    WHERE revision_idempotency_key IS NOT NULL;
"""

# Version 3 — V2.1: occurrences, replay runs, minimized cases.
_V3_SQL = """
CREATE TABLE IF NOT EXISTS fm_occurrences (
    occurrence_id TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL,
    external_trace_id TEXT,
    candidate_id TEXT,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    recurrence_count INTEGER NOT NULL DEFAULT 1,
    environment TEXT,
    agent_name TEXT,
    framework TEXT,
    model_provider TEXT,
    model_name TEXT,
    redacted_metadata_json TEXT NOT NULL DEFAULT '{}',
    severity TEXT NOT NULL DEFAULT 'medium',
    resolution_state TEXT NOT NULL DEFAULT 'open',
    idempotency_key TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fm_occ_fingerprint ON fm_occurrences(fingerprint);
CREATE INDEX IF NOT EXISTS idx_fm_occ_last_seen ON fm_occurrences(last_seen);
CREATE INDEX IF NOT EXISTS idx_fm_occ_resolution ON fm_occurrences(resolution_state);
CREATE UNIQUE INDEX IF NOT EXISTS idx_fm_occ_idempotency
    ON fm_occurrences(idempotency_key) WHERE idempotency_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS fm_replay_runs (
    replay_id TEXT PRIMARY KEY,
    candidate_id TEXT,
    fingerprint TEXT,
    adapter_ref TEXT NOT NULL,
    input_snapshot_json TEXT NOT NULL DEFAULT '{}',
    started_at TEXT NOT NULL,
    ended_at TEXT,
    outcome TEXT NOT NULL,
    failure_category TEXT,
    expected_fingerprint TEXT,
    actual_fingerprint TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    success_count INTEGER NOT NULL DEFAULT 0,
    reproducibility_ratio REAL,
    diagnostics_json TEXT NOT NULL DEFAULT '{}',
    config_json TEXT NOT NULL DEFAULT '{}',
    git_sha TEXT,
    idempotency_key TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fm_replay_candidate ON fm_replay_runs(candidate_id);
CREATE INDEX IF NOT EXISTS idx_fm_replay_fingerprint ON fm_replay_runs(fingerprint);
CREATE UNIQUE INDEX IF NOT EXISTS idx_fm_replay_idempotency
    ON fm_replay_runs(idempotency_key) WHERE idempotency_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS fm_minimized_cases (
    minimization_id TEXT PRIMARY KEY,
    source_candidate_id TEXT NOT NULL,
    source_replay_id TEXT,
    original_size INTEGER NOT NULL,
    minimized_size INTEGER NOT NULL,
    reduction_pct REAL NOT NULL,
    algorithm_version TEXT NOT NULL,
    replay_attempts INTEGER NOT NULL DEFAULT 0,
    reproduction_ratio REAL,
    minimized_payload_json TEXT NOT NULL,
    removed_summary_json TEXT NOT NULL DEFAULT '[]',
    lineage_json TEXT NOT NULL DEFAULT '{}',
    approval_state TEXT NOT NULL DEFAULT 'pending_review',
    exported_at TEXT,
    idempotency_key TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fm_min_candidate ON fm_minimized_cases(source_candidate_id);
CREATE INDEX IF NOT EXISTS idx_fm_min_approval ON fm_minimized_cases(approval_state);
CREATE UNIQUE INDEX IF NOT EXISTS idx_fm_min_idempotency
    ON fm_minimized_cases(idempotency_key) WHERE idempotency_key IS NOT NULL;
"""

MIGRATIONS: tuple[Migration, ...] = (
    Migration(version=1, sql=_V1_SQL, description="initial failure memory schema"),
    Migration(
        version=2,
        sql=_V2_SQL,
        description="candidate lineage and pending-only unique index for revisions",
    ),
    Migration(
        version=3,
        sql=_V3_SQL,
        description="V2.1 occurrences, replay runs, and minimized cases",
    ),
    Migration(
        version=4,
        sql="""
-- Ensure one occurrence row per fingerprint for concurrent writers.
-- Collapse any accidental duplicate fingerprints before unique index.
DELETE FROM fm_occurrences
WHERE occurrence_id NOT IN (
  SELECT occurrence_id FROM (
    SELECT occurrence_id,
           ROW_NUMBER() OVER (
             PARTITION BY fingerprint
             ORDER BY recurrence_count DESC, last_seen DESC
           ) AS rn
    FROM fm_occurrences
  ) ranked WHERE rn = 1
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_fm_occ_fingerprint_unique
    ON fm_occurrences(fingerprint);
""",
        description="unique fingerprint for concurrent occurrence upserts",
    ),
    Migration(
        version=5,
        sql="""
CREATE TABLE IF NOT EXISTS fm_occurrence_deliveries (
    idempotency_key TEXT PRIMARY KEY,
    occurrence_id TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fm_occ_del_fp ON fm_occurrence_deliveries(fingerprint);
""",
        description="idempotent occurrence delivery keys independent of row updates",
    ),
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def configure_connection(conn: sqlite3.Connection, *, busy_timeout_ms: int = 5000) -> None:
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")


def get_schema_version(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='fm_schema_migrations'"
    ).fetchone()
    if row is None:
        return 0
    version_row = conn.execute("SELECT COALESCE(MAX(version), 0) AS v FROM fm_schema_migrations").fetchone()
    return int(version_row["v"] if version_row else 0)


def applied_migrations(conn: sqlite3.Connection) -> list[tuple[int, str]]:
    if get_schema_version(conn) == 0 and not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='fm_schema_migrations'"
    ).fetchone():
        return []
    rows = conn.execute(
        "SELECT version, checksum FROM fm_schema_migrations ORDER BY version"
    ).fetchall()
    return [(int(r["version"]), str(r["checksum"])) for r in rows]


def backup_database(db_path: Path) -> Path:
    """Copy the SQLite database (and WAL/SHM sidecars) before a destructive migration."""
    db_path = Path(db_path)
    if not db_path.is_file():
        raise OSError(f"cannot backup missing database: {db_path}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = db_path.with_name(f"{db_path.name}.bak-{stamp}")
    try:
        shutil.copy2(db_path, backup_path)
        for suffix in ("-wal", "-shm"):
            side = Path(str(db_path) + suffix)
            if side.is_file():
                shutil.copy2(side, Path(str(backup_path) + suffix))
    except OSError as exc:
        raise OSError(f"backup failed for {db_path}: {exc}") from exc
    return backup_path


def restore_database(backup_path: Path, db_path: Path) -> None:
    backup_path = Path(backup_path)
    db_path = Path(db_path)
    if not backup_path.is_file():
        raise OSError(f"backup not found: {backup_path}")
    shutil.copy2(backup_path, db_path)
    for suffix in ("-wal", "-shm"):
        src = Path(str(backup_path) + suffix)
        dst = Path(str(db_path) + suffix)
        if src.is_file():
            shutil.copy2(src, dst)
        elif dst.is_file():
            dst.unlink()


def apply_migrations(
    conn: sqlite3.Connection,
    *,
    db_path: Path | None = None,
    target_version: int | None = None,
    migrations: Iterable[Migration] = MIGRATIONS,
    backup_before_destructive: bool = True,
) -> list[int]:
    """Apply pending migrations up to ``target_version`` (default: latest).

    Returns the list of newly applied versions. Repeat calls are idempotent.
    """
    target = target_version if target_version is not None else CURRENT_SCHEMA_VERSION
    if target < 0:
        raise ValueError("target_version must be >= 0")
    if target > CURRENT_SCHEMA_VERSION:
        raise ValueError(
            f"target_version {target} exceeds CURRENT_SCHEMA_VERSION {CURRENT_SCHEMA_VERSION}"
        )

    migration_list = sorted(migrations, key=lambda m: m.version)
    current = get_schema_version(conn)
    if current > target:
        raise ValueError(
            f"database schema version {current} is newer than requested target {target}; "
            "refusing to silently delete data — restore a backup instead"
        )

    pending = [m for m in migration_list if current < m.version <= target]
    if not pending:
        return []

    # Fresh empty DB (no user tables yet) does not need a backup.
    needs_backup = backup_before_destructive and current > 0 and db_path is not None
    if needs_backup:
        assert db_path is not None
        backup_database(db_path)

    applied: list[int] = []
    for migration in pending:
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.executescript(migration.sql)
            # executescript commits implicitly in some sqlite builds; ensure bookkeeping
            # is recorded in its own transaction if needed.
            existing = conn.execute(
                "SELECT checksum FROM fm_schema_migrations WHERE version = ?",
                (migration.version,),
            ).fetchone()
            if existing is None:
                conn.execute(
                    "INSERT INTO fm_schema_migrations (version, applied_at, checksum) VALUES (?, ?, ?)",
                    (migration.version, _utc_now_iso(), migration.checksum),
                )
            else:
                if str(existing["checksum"]) != migration.checksum:
                    raise RuntimeError(
                        f"migration {migration.version} checksum mismatch: "
                        f"db={existing['checksum']} code={migration.checksum}"
                    )
            conn.commit()
            applied.append(migration.version)
        except Exception:
            try:
                conn.rollback()
            except sqlite3.Error:
                pass
            raise
    return applied


def doctor_report(conn: sqlite3.Connection) -> dict:
    """Return schema health information for ``agenteval memory doctor``."""
    version = get_schema_version(conn)
    applied = applied_migrations(conn)
    expected = {m.version: m.checksum for m in MIGRATIONS if m.version <= CURRENT_SCHEMA_VERSION}
    issues: list[str] = []
    if version == 0:
        issues.append("database is empty (no migrations applied)")
    if version < CURRENT_SCHEMA_VERSION:
        issues.append(
            f"schema is behind: database={version} current={CURRENT_SCHEMA_VERSION}"
        )
    if version > CURRENT_SCHEMA_VERSION:
        issues.append(
            f"schema is newer than this build: database={version} current={CURRENT_SCHEMA_VERSION}"
        )
    for ver, checksum in applied:
        if ver in expected and expected[ver] != checksum:
            issues.append(f"migration {ver} checksum mismatch")
    for ver in expected:
        if ver <= version and ver not in {a[0] for a in applied}:
            issues.append(f"missing migration bookkeeping for version {ver}")

    tables = {
        r["name"]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'fm_%'"
        ).fetchall()
    }
    expected_tables = {
        "fm_schema_migrations",
        "fm_traces",
        "fm_spans",
        "fm_clusters",
        "fm_cluster_members",
        "fm_candidates",
        "fm_review_events",
        "fm_exports",
    }
    if version >= 3:
        expected_tables |= {
            "fm_occurrences",
            "fm_replay_runs",
            "fm_minimized_cases",
        }
    missing_tables = sorted(expected_tables - tables) if version >= 1 else []
    if missing_tables:
        issues.append(f"missing tables: {', '.join(missing_tables)}")

    return {
        "schema_version": version,
        "current_schema_version": CURRENT_SCHEMA_VERSION,
        "applied_migrations": [{"version": v, "checksum": c} for v, c in applied],
        "healthy": not issues,
        "issues": issues,
        "tables": sorted(tables),
    }


def open_migrated_connection(
    db_path: str | Path,
    *,
    busy_timeout_ms: int = 5000,
    create: bool = True,
) -> sqlite3.Connection:
    """Open a SQLite connection and apply pending migrations."""
    path = Path(db_path)
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    elif not path.is_file():
        raise FileNotFoundError(f"Failure Memory database not found: {path}")
    conn = sqlite3.connect(str(path), timeout=busy_timeout_ms / 1000.0, check_same_thread=False)
    configure_connection(conn, busy_timeout_ms=busy_timeout_ms)
    apply_migrations(conn, db_path=path if path.is_file() else None)
    return conn
