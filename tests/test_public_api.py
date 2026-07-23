from __future__ import annotations

import argparse

import pytest

import agenteval
from agenteval.adapters import (
    AgentAdapter,
    AgentResponse,
    AgentRun,
    AutoGenAdapter,
    CrewAIAdapter,
    LangGraphAdapter,
    OpenAIAgentsAdapter,
)
from agenteval.cli import build_parser
from agenteval.compat import AgentEvalDeprecationWarning, warn_deprecated
from agenteval.evaluators import EvaluationContext, EvaluationResult, Evaluator


def test_root_version_and_compatibility_exports():
    assert agenteval.__version__ == "0.1.0"
    assert agenteval.AgentEvalDeprecationWarning is AgentEvalDeprecationWarning
    assert agenteval.warn_deprecated is warn_deprecated


def test_candidate_public_imports_remain_available():
    assert AgentRun is AgentResponse
    assert issubclass(AutoGenAdapter, AgentAdapter)
    assert issubclass(CrewAIAdapter, AgentAdapter)
    assert issubclass(LangGraphAdapter, AgentAdapter)
    assert issubclass(OpenAIAgentsAdapter, AgentAdapter)
    assert EvaluationContext is not None
    assert EvaluationResult is not None
    assert Evaluator is not None


def test_deprecation_warning_is_visible_and_actionable():
    with pytest.warns(
        AgentEvalDeprecationWarning,
        match=r"old_api.*v2\.0.*new_api",
    ):
        warn_deprecated("old_api", removal="v2.0", alternative="new_api")


def test_cli_version_uses_package_version(capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"agenteval {agenteval.__version__}"


def test_existing_and_tier8_cli_commands_are_registered():
    parser = build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    assert set(subparsers.choices) == {
        "run",
        "compare",
        "report",
        "generate",
        "import",
        "generate-cases",
        "init",
        "compare-models",
        "trace",
        "calibrate",
        "audit-log",
        "serve",
        "plugins",
        "templates",
    }
