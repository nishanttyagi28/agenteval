from __future__ import annotations

from pathlib import Path

from agenteval.cli import (
    _cmd_templates_install,
    _cmd_templates_list,
    _cmd_templates_show,
    build_parser,
)


def parse(*argv):
    return build_parser().parse_args(list(argv))


def test_template_commands_are_registered():
    assert parse("templates", "list").func is _cmd_templates_list
    assert parse("templates", "show", "rag-assistant").func is _cmd_templates_show
    assert (
        parse("templates", "install", "coding-agent").func
        is _cmd_templates_install
    )


def test_templates_list_output(capsys):
    assert _cmd_templates_list(parse("templates", "list")) == 0
    output = capsys.readouterr().out
    assert "rag-assistant" in output
    assert "coding-agent" in output
    assert "customer-support" in output
    assert "bundled" in output


def test_templates_show_output(capsys):
    assert _cmd_templates_show(parse("templates", "show", "customer-support")) == 0
    output = capsys.readouterr().out
    assert "Customer-Support Agent" in output
    assert "account_privacy_boundary" in output


def test_templates_show_unknown(capsys):
    assert _cmd_templates_show(parse("templates", "show", "missing")) == 2
    assert "Unknown template" in capsys.readouterr().err


def test_templates_install_to_explicit_output(tmp_path, capsys):
    destination = tmp_path / "installed"
    args = parse(
        "templates",
        "install",
        "rag-assistant",
        "--output",
        str(destination),
    )
    assert _cmd_templates_install(args) == 0
    assert (destination / "agents.yaml").is_file()
    assert (destination / "cases.yaml").is_file()
    assert str(destination.resolve()) in capsys.readouterr().out


def test_templates_install_refuses_overwrite_then_allows_force(tmp_path, capsys):
    destination = tmp_path / "installed"
    destination.mkdir()
    target = destination / "README.md"
    target.write_text("user file\n", encoding="utf-8")

    args = parse(
        "templates",
        "install",
        "coding-agent",
        "--output",
        str(destination),
    )
    assert _cmd_templates_install(args) == 1
    assert target.read_text(encoding="utf-8") == "user file\n"
    assert "--force" in capsys.readouterr().err

    forced = parse(
        "templates",
        "install",
        "coding-agent",
        "--output",
        str(destination),
        "--force",
    )
    assert _cmd_templates_install(forced) == 0
    assert target.read_text(encoding="utf-8").startswith("# Coding agent")
