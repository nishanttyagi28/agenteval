from datetime import datetime, timezone

from agenteval.core.audit import AuditEntry, append_audit_entry, build_entry, read_audit_log


def test_build_entry_stamps_utc_timestamp_and_defaults():
    entry = build_entry("run", details={"run_id": "r1"})
    assert entry.action == "run"
    assert entry.actor == "local"
    assert entry.outcome == "ok"
    assert entry.details == {"run_id": "r1"}
    parsed = datetime.fromisoformat(entry.timestamp)
    assert parsed.tzinfo is not None


def test_build_entry_accepts_custom_actor_and_outcome():
    entry = build_entry("compare", actor="ci", outcome="failed")
    assert entry.actor == "ci"
    assert entry.outcome == "failed"
    assert entry.details == {}


# ── append_audit_entry / read_audit_log round trip ──────────────────────────


def test_append_and_read_round_trip(tmp_path):
    path = tmp_path / "audit.jsonl"
    append_audit_entry(build_entry("run", details={"n": 1}), path)
    append_audit_entry(build_entry("compare", details={"n": 2}), path)

    entries = read_audit_log(path)
    assert len(entries) == 2
    assert entries[0].action == "run"
    assert entries[1].action == "compare"


def test_append_creates_parent_directory(tmp_path):
    path = tmp_path / "nested" / "dir" / "audit.jsonl"
    append_audit_entry(build_entry("run"), path)
    assert path.is_file()


def test_read_audit_log_missing_file_returns_empty_list(tmp_path):
    assert read_audit_log(tmp_path / "nope.jsonl") == []


def test_read_audit_log_skips_corrupted_lines(tmp_path):
    path = tmp_path / "audit.jsonl"
    path.write_text('{"bad json\nnot even json at all\n', encoding="utf-8")
    append_audit_entry(build_entry("run"), path)

    entries = read_audit_log(path)
    assert len(entries) == 1


def test_read_audit_log_skips_non_object_lines(tmp_path):
    path = tmp_path / "audit.jsonl"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")
    append_audit_entry(build_entry("run"), path)

    entries = read_audit_log(path)
    assert len(entries) == 1


def test_read_audit_log_ignores_blank_lines(tmp_path):
    path = tmp_path / "audit.jsonl"
    append_audit_entry(build_entry("run"), path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n\n   \n")

    entries = read_audit_log(path)
    assert len(entries) == 1


# ── since filtering ──────────────────────────────────────────────────────────


def write_entry_at(path, timestamp: str, action: str):
    entry = AuditEntry(timestamp=timestamp, actor="local", action=action, details={}, outcome="ok")
    append_audit_entry(entry, path)


def test_since_filters_out_earlier_entries(tmp_path):
    path = tmp_path / "audit.jsonl"
    write_entry_at(path, "2026-01-01T00:00:00+00:00", "old")
    write_entry_at(path, "2026-06-01T00:00:00+00:00", "new")

    entries = read_audit_log(path, since=datetime(2026, 3, 1, tzinfo=timezone.utc))
    assert [entry.action for entry in entries] == ["new"]


def test_since_is_inclusive_of_exact_match(tmp_path):
    path = tmp_path / "audit.jsonl"
    write_entry_at(path, "2026-03-01T00:00:00+00:00", "boundary")

    entries = read_audit_log(path, since=datetime(2026, 3, 1, tzinfo=timezone.utc))
    assert len(entries) == 1


def test_since_naive_datetime_is_treated_as_utc(tmp_path):
    path = tmp_path / "audit.jsonl"
    write_entry_at(path, "2026-06-01T00:00:00+00:00", "new")

    # Naive `since` (no tzinfo) must still compare correctly against UTC entries.
    entries = read_audit_log(path, since=datetime(2026, 3, 1))
    assert len(entries) == 1


def test_since_excludes_entries_with_unparsable_timestamp(tmp_path):
    path = tmp_path / "audit.jsonl"
    write_entry_at(path, "not-a-real-timestamp", "bad_ts")
    write_entry_at(path, "2026-06-01T00:00:00+00:00", "good_ts")

    entries = read_audit_log(path, since=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert [entry.action for entry in entries] == ["good_ts"]


def test_no_since_returns_all_entries_regardless_of_timestamp_validity(tmp_path):
    path = tmp_path / "audit.jsonl"
    write_entry_at(path, "not-a-real-timestamp", "bad_ts")
    write_entry_at(path, "2026-06-01T00:00:00+00:00", "good_ts")

    entries = read_audit_log(path)
    assert len(entries) == 2
