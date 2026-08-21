"""Screenshot-ready terminal demo of the Indic-language evaluation pack.

Runs, in order: template install, filtered plugin discovery, and the full
"core" (deterministic, offline) eval against the bundled mock agent. Output
is intentionally compact (~45 lines) for a LinkedIn screenshot -- verbose
per-case logging (latency, cost, running/done pairs) is trimmed; the result
line and scoring mechanism for every case is kept.

Zero network calls, no API key needed: the three custom checkers are wired
into the evaluator registry the same lightweight way
tests/test_indic_mock_agent_demo.py does (sys.path + a fake entry point --
the same evaluate() functions a real ``pip install`` would run), so this
works right after the repo's own documented ``pip install -e ".[dev]"``
with no extra setup.

Usage (from the repo root):
    python scripts/demo_indic.py
"""

from __future__ import annotations

import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
INDIC_SRC = ROOT / "examples" / "plugins" / "agenteval-indic-evaluators" / "src"
if str(INDIC_SRC) not in sys.path:
    sys.path.insert(0, str(INDIC_SRC))

from agenteval.core.registry import load_adapter_class  # noqa: E402
from agenteval.core.runner import run_golden_suite  # noqa: E402
from agenteval.core.schema import load_test_cases  # noqa: E402
from agenteval.core.template_catalog import install_template  # noqa: E402
from agenteval.evaluators._registry import discover_evaluators  # noqa: E402

WIDTH = 66
DEMO_DIR = ROOT / "examples" / "indic_mock_agent"


def _register_indic_evaluators() -> None:
    """Resolve the 3 checkers through the real registry without a pip
    install: same evaluate() functions, same agenteval.evaluators._registry
    code path, just discovered via an in-process fake entry point."""
    from agenteval_indic_evaluators.script_consistency import evaluate as sc
    from agenteval_indic_evaluators.tool_arg_encoding import evaluate as tae
    from agenteval_indic_evaluators.transliteration_stability import (
        evaluate as ts,
    )

    @dataclass
    class _Dist:
        name: str = "agenteval-indic-evaluators"
        version: str = "0.1.0"

        @property
        def metadata(self) -> dict[str, str]:
            return {"Name": self.name}

    class _Entry:
        def __init__(self, name: str, value: str, plugin: Any) -> None:
            self.name = name
            self.value = value
            self.group = "agenteval.evaluators"
            self.dist = _Dist()
            self._plugin = plugin

        def load(self) -> Any:
            return self._plugin

    entries = [
        _Entry("script_consistency", "agenteval_indic_evaluators.script_consistency:evaluate", sc),
        _Entry("transliteration_stability", "agenteval_indic_evaluators.transliteration_stability:evaluate", ts),
        _Entry("tool_arg_encoding", "agenteval_indic_evaluators.tool_arg_encoding:evaluate", tae),
    ]
    import agenteval.evaluators._registry as registry

    registry._installed_entry_points = lambda: entries


def _rule(char: str = "=") -> str:
    return char * WIDTH


def _short_reason(checker: str, reason: str) -> str:
    """Compress a real evaluator reason string into one screenshot line."""
    reason = reason or ""
    if checker == "numeric":
        m = re.search(r"expected .\s*([\d.]+).*found \[([\d.]+)", reason)
        if m:
            return f"expected ~{float(m.group(1)):.0f}, got {float(m.group(2)):.0f}"
    elif checker == "script_consistency":
        m = re.search(r"(\d+)% .* outside the expected (\w+) script", reason)
        if m:
            return f"{m.group(1)}% of reply outside {m.group(2)} script"
    elif checker == "transliteration_stability":
        m = re.search(r"\[(.*?)\] used interchangeably", reason)
        if m:
            return f"used both {m.group(1)} for one entity"
    elif checker == "tool_arg_encoding":
        if "mangled encoding" in reason:
            return "tool argument arrived mangled"
        if "did not contain" in reason:
            return "tool argument missing expected text"
    return (reason[:44] + "...") if len(reason) > 44 else reason


def main() -> int:
    _register_indic_evaluators()

    print(_rule())
    print("  AgentEval -- Indic-Language Evaluation Pack (offline demo)")
    print(_rule())

    tmp_root = Path(tempfile.mkdtemp(prefix="agenteval-indic-demo-"))
    try:
        print()
        print("[1] agenteval templates install indic-agent")
        dest = tmp_root / "indic-agent"
        written = install_template("indic-agent", dest)
        installed_cases = load_test_cases(dest / "cases.yaml")
        print(f"    -> installed to <tmp>/indic-agent  ({len(installed_cases)} cases, {len(written)} files)")

        print()
        print("[2] agenteval plugins list   (indic evaluators only)")
        for info in discover_evaluators():
            if info.package == "agenteval-indic-evaluators":
                print(f"    {info.name:<28} {info.package}  {info.version}")

        print()
        print("[3] agenteval run --agent indic_agent --tag core --no-llm-judge")
        adapter_cls = load_adapter_class(
            "examples.indic_mock_agent.adapter:MockAgentAdapterIndic"
        )
        adapter = adapter_cls(repo_path=DEMO_DIR)
        cases = load_test_cases(DEMO_DIR / "cases.yaml")
        checker_by_case = {
            c.id: (c.expects.evaluator or c.expects.correctness_type.value) for c in cases
        }

        report = run_golden_suite(
            adapter,
            cases_path=DEMO_DIR / "cases.yaml",
            adapter_name="indic_agent",
            tags=["core"],
            verbose=False,
            score=True,
            use_llm_judge=False,
        )

        passed = 0
        failed_by_checker: dict[str, int] = {}
        for r in report.case_results:
            checker = checker_by_case.get(r.case_id, "?")
            if r.correctness_pass:
                passed += 1
                print(f"    PASS  {r.case_id:<34} {checker}")
            else:
                failed_by_checker[checker] = failed_by_checker.get(checker, 0) + 1
                reason = _short_reason(checker, r.judge_reason or "")
                print(f"    FAIL  {r.case_id:<34} {checker:<26} {reason}")

        total = len(report.case_results)
        failed = total - passed
        errors = report.evaluator_error_count + report.agent_error_count

        print()
        print(_rule())
        print(f"  {total} cases run  |  {passed} passed  |  {failed} failed-by-design  |  {errors} errors")
        by_checker = "  ".join(f"{k}({v})" for k, v in failed_by_checker.items())
        print(f"  Failures caught by: {by_checker}")
        print("  Network calls: 0    API key required: no")
        print(_rule())
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
