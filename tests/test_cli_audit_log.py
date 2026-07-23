from pathlib import Path

from agenteval.cli import _cmd_audit_log, build_parser
from agenteval.core.audit import append_audit_entry, build_entry
from agenteval.core.schema import AgentConfig, GateConfig, RepositoryConfig


def make_config(tmp_path: Path, name: str = "example_agent", *, log_path=None) -> AgentConfig:
    from agenteval.core.schema import AuditConfig

    return AgentConfig(
        name=name,
        display_name="Example Agent",
        adapter="agenteval.adapters.scheme_saathi:SchemeSaathiAdapter",
        repository=RepositoryConfig(env_var=f"{name.upper()}_PATH"),
        golden_suite=tmp_path / "golden.yaml",
        baseline=tmp_path / "baseline.json",
        runs_dir=Path("runs"),
        gates=GateConfig(),
        audit=AuditConfig(enabled=True, log_path=log_path),
    )


def setup_registry(tmp_path, monkeypatch, config=None):
    config = config or make_config(tmp_path)
    monkeypatch.setattr(
        "agenteval.core.registry.load_agent_registry", lambda path: {config.name: config}
    )
    return config


def parse(argv):
    return build_parser().parse_args(["audit-log", *argv])


def test_audit_log_reads_default_sidecar_path(tmp_path, monkeypatch, capsys):
    setup_registry(tmp_path, monkeypatch)
    log_path = tmp_path / "runs" / "example_agent" / "audit.jsonl"
    append_audit_entry(build_entry("run", details={"run_id": "r1"}), log_path)

    args = parse(["--agent", "example_agent", "--registry", str(tmp_path / "agents.yaml")])
    assert _cmd_audit_log(args) == 0

    out = capsys.readouterr().out
    assert "entries=1" in out
    assert "action=run" in out


def test_audit_log_respects_custom_log_path(tmp_path, monkeypatch, capsys):
    config = make_config(tmp_path, log_path="custom/audit.jsonl")
    setup_registry(tmp_path, monkeypatch, config)
    log_path = tmp_path / "custom" / "audit.jsonl"
    append_audit_entry(build_entry("compare"), log_path)

    args = parse(["--agent", "example_agent", "--registry", str(tmp_path / "agents.yaml")])
    assert _cmd_audit_log(args) == 0
    assert "action=compare" in capsys.readouterr().out


def test_audit_log_since_filters_entries(tmp_path, monkeypatch, capsys):
    setup_registry(tmp_path, monkeypatch)
    log_path = tmp_path / "runs" / "example_agent" / "audit.jsonl"
    from agenteval.core.audit import AuditEntry

    append_audit_entry(
        AuditEntry(timestamp="2026-01-01T00:00:00+00:00", actor="local", action="old", details={}),
        log_path,
    )
    append_audit_entry(
        AuditEntry(timestamp="2026-06-01T00:00:00+00:00", actor="local", action="new", details={}),
        log_path,
    )

    args = parse(
        ["--agent", "example_agent", "--registry", str(tmp_path / "agents.yaml"), "--since", "2026-03-01"]
    )
    assert _cmd_audit_log(args) == 0
    out = capsys.readouterr().out
    assert "entries=1" in out
    assert "action=new" in out
    assert "action=old" not in out


def test_audit_log_no_entries_yet(tmp_path, monkeypatch, capsys):
    setup_registry(tmp_path, monkeypatch)
    args = parse(["--agent", "example_agent", "--registry", str(tmp_path / "agents.yaml")])
    assert _cmd_audit_log(args) == 0
    assert "entries=0" in capsys.readouterr().out


def test_audit_log_unknown_agent_is_a_clean_error(tmp_path, monkeypatch, capsys):
    setup_registry(tmp_path, monkeypatch)
    args = parse(["--agent", "does_not_exist", "--registry", str(tmp_path / "agents.yaml")])
    assert _cmd_audit_log(args) == 2
    assert "Unknown agent" in capsys.readouterr().err


def test_audit_log_invalid_since_is_a_clean_error(tmp_path, monkeypatch, capsys):
    setup_registry(tmp_path, monkeypatch)
    args = parse(
        ["--agent", "example_agent", "--registry", str(tmp_path / "agents.yaml"), "--since", "not-a-date"]
    )
    assert _cmd_audit_log(args) == 2
    assert "ISO-8601" in capsys.readouterr().err
