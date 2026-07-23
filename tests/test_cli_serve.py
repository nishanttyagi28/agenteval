from pathlib import Path

from agenteval.cli import _cmd_serve, build_parser
from agenteval.core.schema import AgentConfig, GateConfig, RepositoryConfig


class _StubServer:
    def __init__(self, host: str, port: int) -> None:
        self.server_address = (host, port or 8765)
        self.serve_forever_called = False
        self.shutdown_called = False
        self.closed = False

    def serve_forever(self) -> None:
        self.serve_forever_called = True

    def shutdown(self) -> None:
        self.shutdown_called = True

    def server_close(self) -> None:
        self.closed = True


def make_config(tmp_path: Path, name: str = "example_agent") -> AgentConfig:
    return AgentConfig(
        name=name,
        display_name="Example Agent",
        adapter="agenteval.adapters.scheme_saathi:SchemeSaathiAdapter",
        repository=RepositoryConfig(env_var=f"{name.upper()}_PATH"),
        golden_suite=tmp_path / "golden.yaml",
        baseline=tmp_path / "baseline.json",
        runs_dir=Path("runs"),
        gates=GateConfig(),
    )


def setup_registry(tmp_path, monkeypatch, configs=None):
    configs = configs or [make_config(tmp_path)]
    registry = {config.name: config for config in configs}
    monkeypatch.setattr("agenteval.core.registry.load_agent_registry", lambda path: registry)
    return registry


def parse(argv):
    return build_parser().parse_args(["serve", *argv])


def test_serve_requires_local_flag(tmp_path, monkeypatch, capsys):
    setup_registry(tmp_path, monkeypatch)
    args = parse(["--registry", str(tmp_path / "agents.yaml")])
    assert _cmd_serve(args) == 2
    assert "--local is required" in capsys.readouterr().err


def test_serve_happy_path_defaults_to_all_agents(tmp_path, monkeypatch, capsys):
    setup_registry(tmp_path, monkeypatch)
    captured = {}

    def fake_run_server(agent_paths, *, host, port):
        captured["agent_paths"] = agent_paths
        captured["host"] = host
        captured["port"] = port
        return _StubServer(host, port)

    monkeypatch.setattr("agenteval.core.server.run_server", fake_run_server)

    args = parse(["--local", "--registry", str(tmp_path / "agents.yaml")])
    assert _cmd_serve(args) == 0

    assert "example_agent" in captured["agent_paths"]
    out = capsys.readouterr().out
    assert "agents=example_agent" in out
    assert "serving on http://127.0.0.1:8765" in out


def test_serve_specific_agent_only(tmp_path, monkeypatch, capsys):
    configs = [make_config(tmp_path, "agent_a"), make_config(tmp_path, "agent_b")]
    setup_registry(tmp_path, monkeypatch, configs)
    captured = {}

    def fake_run_server(agent_paths, *, host, port):
        captured["agent_paths"] = agent_paths
        return _StubServer(host, port)

    monkeypatch.setattr("agenteval.core.server.run_server", fake_run_server)

    args = parse(["--local", "--agent", "agent_a", "--registry", str(tmp_path / "agents.yaml")])
    assert _cmd_serve(args) == 0
    assert set(captured["agent_paths"]) == {"agent_a"}


def test_serve_unknown_agent_is_a_clean_error(tmp_path, monkeypatch, capsys):
    setup_registry(tmp_path, monkeypatch)
    args = parse(["--local", "--agent", "does_not_exist", "--registry", str(tmp_path / "agents.yaml")])
    assert _cmd_serve(args) == 2
    assert "Unknown agent" in capsys.readouterr().err


def test_serve_custom_host_and_port(tmp_path, monkeypatch, capsys):
    setup_registry(tmp_path, monkeypatch)

    def fake_run_server(agent_paths, *, host, port):
        return _StubServer(host, port)

    monkeypatch.setattr("agenteval.core.server.run_server", fake_run_server)

    args = parse(
        ["--local", "--host", "0.0.0.0", "--port", "9000", "--registry", str(tmp_path / "agents.yaml")]
    )
    assert _cmd_serve(args) == 0
    assert "serving on http://0.0.0.0:9000" in capsys.readouterr().out


def test_serve_shuts_down_cleanly_after_serve_forever(tmp_path, monkeypatch):
    setup_registry(tmp_path, monkeypatch)
    stub_holder = {}

    def fake_run_server(agent_paths, *, host, port):
        stub = _StubServer(host, port)
        stub_holder["stub"] = stub
        return stub

    monkeypatch.setattr("agenteval.core.server.run_server", fake_run_server)

    args = parse(["--local", "--registry", str(tmp_path / "agents.yaml")])
    assert _cmd_serve(args) == 0
    assert stub_holder["stub"].serve_forever_called
    assert stub_holder["stub"].shutdown_called
    assert stub_holder["stub"].closed
