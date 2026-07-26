"""CLI help text should stay framework-neutral (Issue 15)."""

from __future__ import annotations

from agenteval.cli import build_parser


def test_run_agent_repo_help_is_framework_neutral():
    parser = build_parser()
    run_parser = None
    for action in parser._subparsers._group_actions:  # noqa: SLF001
        choices = getattr(action, "choices", None) or {}
        if "run" in choices:
            run_parser = choices["run"]
            break
    assert run_parser is not None
    help_text = run_parser.format_help()
    assert "Agentic Data Analyst root" not in help_text
    assert "agent repository under test" in help_text


def test_top_level_help_lists_run_and_compare():
    help_text = build_parser().format_help()
    assert "run" in help_text
    assert "compare" in help_text
