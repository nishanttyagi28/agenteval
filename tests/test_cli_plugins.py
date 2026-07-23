from __future__ import annotations

from importlib import metadata
from pathlib import Path

import pytest

from agenteval.cli import (
    _cmd_plugins_inspect,
    _cmd_plugins_list,
    _cmd_plugins_validate,
    build_parser,
)
from agenteval.core.schema import CaseResult, Expects, TestCase
from agenteval.evaluators import EvaluationContext
from agenteval.evaluators._registry import evaluate


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PLUGIN = ROOT / "examples" / "plugins" / "agenteval-keyword-evaluator"


class FakeDistribution:
    version = "2.0.0"
    metadata = {"Name": "third-party-evaluator"}


class FakeEntryPoint:
    name = "third_party"
    value = "third_party:evaluate"
    group = "agenteval.evaluators"
    dist = FakeDistribution()

    def __init__(self, plugin):
        self.plugin = plugin
        self.loaded = False

    def load(self):
        self.loaded = True
        return self.plugin


class EntryPoints(list):
    def select(self, **params):
        if params.get("group") == "agenteval.evaluators":
            return EntryPoints(self)
        return EntryPoints()


def parse(*argv):
    return build_parser().parse_args(list(argv))


def test_plugins_commands_are_registered():
    assert parse("plugins", "list").func is _cmd_plugins_list
    assert parse("plugins", "inspect", "exact").func is _cmd_plugins_inspect
    assert parse("plugins", "validate", "exact").func is _cmd_plugins_validate


def test_plugins_list_does_not_load_third_party_code(monkeypatch, capsys):
    entry_point = FakeEntryPoint(lambda context: None)
    monkeypatch.setattr(
        "agenteval.evaluators._registry.metadata.entry_points",
        lambda: EntryPoints([entry_point]),
    )
    assert _cmd_plugins_list(parse("plugins", "list")) == 0
    captured = capsys.readouterr()
    assert "third_party" in captured.out
    assert "third-party-evaluator" in captured.out
    assert "discovered" in captured.out
    assert entry_point.loaded is False


def test_plugins_list_reports_metadata_errors(monkeypatch, capsys):
    entry_point = FakeEntryPoint(lambda context: None)
    entry_point.name = "Bad Name"
    monkeypatch.setattr(
        "agenteval.evaluators._registry.metadata.entry_points",
        lambda: EntryPoints([entry_point]),
    )
    assert _cmd_plugins_list(parse("plugins", "list")) == 1
    captured = capsys.readouterr()
    assert "malformed" in captured.out
    assert "invalid evaluator name" in captured.err
    assert entry_point.loaded is False


def test_plugins_list_reports_entry_point_backend_failure(monkeypatch, capsys):
    def fail():
        raise RuntimeError("metadata backend failed")

    monkeypatch.setattr(
        "agenteval.evaluators._registry.metadata.entry_points",
        fail,
    )
    assert _cmd_plugins_list(parse("plugins", "list")) == 1
    assert "metadata backend failed" in capsys.readouterr().err


def test_plugins_inspect_is_metadata_only(monkeypatch, capsys):
    entry_point = FakeEntryPoint(lambda context: None)
    monkeypatch.setattr(
        "agenteval.evaluators._registry.metadata.entry_points",
        lambda: EntryPoints([entry_point]),
    )
    assert _cmd_plugins_inspect(parse("plugins", "inspect", "third_party")) == 0
    captured = capsys.readouterr()
    assert "Loaded: no" in captured.out
    assert "Target: third_party:evaluate" in captured.out
    assert entry_point.loaded is False


def test_plugins_validate_loads_but_does_not_invoke(monkeypatch, capsys):
    invoked = False

    def plugin(_context):
        nonlocal invoked
        invoked = True

    entry_point = FakeEntryPoint(plugin)
    monkeypatch.setattr(
        "agenteval.evaluators._registry.metadata.entry_points",
        lambda: EntryPoints([entry_point]),
    )
    assert _cmd_plugins_validate(parse("plugins", "validate", "third_party")) == 0
    captured = capsys.readouterr()
    assert "The evaluator callable was not executed" in captured.out
    assert entry_point.loaded is True
    assert invoked is False


def test_plugins_validate_unknown_name(monkeypatch, capsys):
    monkeypatch.setattr(
        "agenteval.evaluators._registry.metadata.entry_points",
        lambda: EntryPoints(),
    )
    assert _cmd_plugins_validate(parse("plugins", "validate", "missing")) == 2
    assert "unknown evaluator" in capsys.readouterr().err


def test_plugins_validate_reports_builtin_name_collision(monkeypatch, capsys):
    entry_point = FakeEntryPoint(lambda context: None)
    entry_point.name = "exact"
    monkeypatch.setattr(
        "agenteval.evaluators._registry.metadata.entry_points",
        lambda: EntryPoints([entry_point]),
    )
    assert _cmd_plugins_validate(parse("plugins", "validate", "exact")) == 1
    assert "reserved by a built-in evaluator" in capsys.readouterr().err


def test_example_plugin_declares_expected_entry_point():
    with (EXAMPLE_PLUGIN / "pyproject.toml").open("rb") as handle:
        config = __import__("tomllib").load(handle)
    assert config["project"]["entry-points"]["agenteval.evaluators"] == {
        "keyword_contains": "agenteval_keyword_evaluator:evaluate"
    }


def test_example_plugin_loads_and_evaluates_through_real_entry_point(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE_PLUGIN / "src"))
    entry_point = metadata.EntryPoint(
        name="keyword_contains",
        value="agenteval_keyword_evaluator:evaluate",
        group="agenteval.evaluators",
    )
    case = TestCase(
        id="refund",
        prompt="Explain the refund policy",
        expects=Expects.from_dict(
            {"evaluator": "keyword_contains", "ground_truth": "30 days"}
        ),
    )
    result = evaluate(
        "keyword_contains",
        EvaluationContext(
            case=case,
            result=CaseResult(
                case_id=case.id,
                prompt=case.prompt,
                final_answer="Refunds are available within 30 days.",
            ),
        ),
        [entry_point],
    )
    assert result.passed is True
    assert result.reason == "found keyword '30 days'"
