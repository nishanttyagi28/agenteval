"""CrewAI adapter that normalizes crew execution into AgentEval responses.

CrewAI is deliberately an optional dependency.  The adapter works against the
documented ``Crew.kickoff``/``CrewOutput`` interface and uses duck typing so
importing AgentEval does not require CrewAI to be installed.
"""

from __future__ import annotations

import importlib
import json
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from agenteval.adapters.base import AgentAdapter, AgentResponse


def _value(source: Any, *names: str) -> Any:
    """Read the first present mapping key or object attribute."""
    for name in names:
        if isinstance(source, Mapping) and name in source:
            value = source[name]
            if value is not None:
                return value
            continue
        try:
            value = getattr(source, name)
        except (AttributeError, TypeError):
            continue
        if value is not None:
            return value
    return None


def _plain(value: Any) -> Any:
    """Convert CrewAI/Pydantic values into JSON-safe evidence."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _plain(model_dump(mode="json"))
        except Exception:  # noqa: BLE001 - evidence capture must not break a crew
            try:
                return _plain(model_dump())
            except Exception:  # noqa: BLE001
                pass
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _plain(to_dict())
        except Exception:  # noqa: BLE001 - fall back to a stable string
            pass
    try:
        return str(value)
    except Exception:  # noqa: BLE001
        return f"<{type(value).__name__}>"


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _name(value: Any) -> str | None:
    """Normalize a tool/agent/task name without changing its public spelling."""
    if value is None:
        return None
    if not isinstance(value, str):
        value = _value(value, "name", "role") or value
    result = str(value).strip()
    return result or None


def _tool_call_names(value: Any) -> list[str]:
    """Extract tool names from OpenAI-style tool call collections."""
    if value is None:
        return []
    calls = value if isinstance(value, (list, tuple)) else [value]
    names: list[str] = []
    for call in calls:
        function = _value(call, "function")
        candidate = _value(function, "name") if function is not None else None
        candidate = candidate or _value(call, "name", "tool", "tool_name")
        normalized = _name(candidate)
        if normalized:
            names.append(normalized)
    return names


def _tools_from_step(step: Any) -> list[str]:
    """Extract CrewAI ``AgentAction.tool`` values from a step callback payload."""
    if isinstance(step, (list, tuple)):
        return [name for item in step for name in _tools_from_step(item)]
    names = _tool_call_names(_value(step, "tool_calls"))
    direct = _name(_value(step, "tool", "tool_name"))
    if direct:
        names.insert(0, direct)
    return names


def _tools_from_messages(messages: Any) -> list[str]:
    if not isinstance(messages, Iterable) or isinstance(messages, (str, bytes, Mapping)):
        return []
    names: list[str] = []
    for message in messages:
        names.extend(_tool_call_names(_value(message, "tool_calls")))
        if str(_value(message, "role") or "").lower() == "tool":
            tool_name = _name(_value(message, "name", "tool_name"))
            if tool_name:
                names.append(tool_name)
    return names


@contextmanager
def _capture_steps(crew: Any):
    """Temporarily chain CrewAI's crew-level callback and collect observed steps."""
    steps: list[Any] = []
    tool_calls: list[str] = []
    try:
        original = getattr(crew, "step_callback")
    except AttributeError:
        original = None

    def capture(step: Any) -> None:
        try:
            steps.append(_plain(step))
            tool_calls.extend(_tools_from_step(step))
        except Exception:  # noqa: BLE001 - observability cannot change execution
            steps.append(f"<{type(step).__name__}>")
        if callable(original):
            original(step)

    installed = False
    try:
        setattr(crew, "step_callback", capture)
        installed = True
    except (AttributeError, TypeError, ValueError):
        # Some wrappers expose a read-only Crew facade. Task messages still
        # provide a non-invasive source of tool-call evidence in that case.
        pass
    try:
        yield steps, tool_calls
    finally:
        if installed:
            try:
                setattr(crew, "step_callback", original)
            except (AttributeError, TypeError, ValueError):
                pass


def _complete_stream(output: Any) -> Any:
    """Consume CrewStreamingOutput and return its final CrewOutput."""
    class_name = type(output).__name__
    if "StreamingOutput" not in class_name:
        return output
    for _chunk in output:
        pass
    final = getattr(output, "result", None)
    if final is None:
        raise RuntimeError("CrewAI streaming execution completed without a final result")
    return final


def _final_answer(output: Any) -> str:
    raw = _value(output, "raw")
    if raw is not None and str(raw).strip():
        return str(raw).strip()
    structured = _value(output, "pydantic", "json_dict")
    if structured is not None:
        return json.dumps(_plain(structured), ensure_ascii=False, sort_keys=True)
    return str(output).strip() if output is not None else ""


def _task_outputs(output: Any, crew: Any) -> list[Any]:
    tasks_output = _value(output, "tasks_output")
    if tasks_output is not None:
        return list(tasks_output)
    results: list[Any] = []
    for task in list(getattr(crew, "tasks", None) or []):
        task_output = _value(task, "output")
        if task_output is not None:
            results.append(task_output)
    return results


def _trajectory(task_outputs: list[Any], crew: Any) -> list[str]:
    configured_tasks = list(getattr(crew, "tasks", None) or [])
    nodes: list[str] = []
    for index, task_output in enumerate(task_outputs):
        configured = configured_tasks[index] if index < len(configured_tasks) else None
        task_name = _name(_value(task_output, "name"))
        task_name = task_name or _name(_value(configured, "name")) or str(index + 1)
        nodes.append(f"task:{task_name}")

        agent_name = _name(_value(task_output, "agent"))
        if agent_name is None and configured is not None:
            agent_name = _name(_value(_value(configured, "agent"), "role", "name"))
        if agent_name:
            nodes.append(f"agent:{agent_name}")
    return nodes


class CrewAIAdapter(AgentAdapter):
    """Run a CrewAI crew and expose output, tools, trajectory, and usage.

    Pass a reusable ``crew`` instance, a ``crew_factory``, or a CLI-friendly
    ``crew_import`` such as ``"my_project.crew:MyProjectCrew"``. Imported
    CrewBase-style classes are instantiated and their ``crew`` method is called
    for every case. The case prompt is supplied to ``kickoff`` under
    ``input_key`` (``prompt`` by default); keyword arguments passed to
    :meth:`run` become extra inputs.
    """

    def __init__(
        self,
        crew: Any | None = None,
        *,
        crew_factory: Callable[[], Any] | None = None,
        crew_import: str | None = None,
        crew_method: str | None = "crew",
        repo_path: str | Path | None = None,
        input_key: str = "prompt",
        inputs: Mapping[str, Any] | None = None,
    ) -> None:
        sources = sum(value is not None for value in (crew, crew_factory, crew_import))
        if sources != 1:
            raise ValueError("provide exactly one of crew, crew_factory, or crew_import")
        if not isinstance(input_key, str) or not input_key.strip():
            raise ValueError("input_key must be a non-empty string")
        if crew is not None and not callable(getattr(crew, "kickoff", None)):
            raise TypeError("crew must provide a callable kickoff method")
        if crew_factory is not None and not callable(crew_factory):
            raise TypeError("crew_factory must be callable")
        if crew_import is not None:
            if not isinstance(crew_import, str) or crew_import.count(":") != 1:
                raise ValueError("crew_import must use 'module.path:Name' format")
            module_name, symbol_name = crew_import.split(":", 1)
            if not module_name.strip() or not symbol_name.strip():
                raise ValueError("crew_import must use 'module.path:Name' format")
        if crew_method is not None and (
            not isinstance(crew_method, str) or not crew_method.strip()
        ):
            raise ValueError("crew_method must be a non-empty string or None")
        self.crew = crew
        self.crew_factory = crew_factory
        self.crew_import = crew_import
        self.crew_method = crew_method.strip() if isinstance(crew_method, str) else None
        self.repo_path = Path(repo_path).expanduser().resolve() if repo_path else None
        self.input_key = input_key.strip()
        self.inputs = dict(inputs or {})

    def _imported_crew(self) -> Any:
        if self.crew_import is None:
            raise RuntimeError("crew_import is not configured")
        if self.repo_path is not None:
            for candidate in (self.repo_path, self.repo_path / "src"):
                candidate_text = str(candidate)
                if candidate.is_dir() and candidate_text not in sys.path:
                    sys.path.insert(0, candidate_text)
        module_name, symbol_name = self.crew_import.split(":", 1)
        try:
            target = getattr(importlib.import_module(module_name), symbol_name)
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                f"cannot import CrewAI entrypoint {self.crew_import!r}: {exc}"
            ) from exc
        instance = target()
        if self.crew_method is None:
            return instance
        factory = getattr(instance, self.crew_method, None)
        if not callable(factory):
            raise TypeError(
                f"CrewAI entrypoint {self.crew_import!r} has no callable "
                f"{self.crew_method!r} method"
            )
        return factory()

    def _crew_for_run(self) -> Any:
        if self.crew_import is not None:
            crew = self._imported_crew()
        else:
            crew = self.crew_factory() if self.crew_factory is not None else self.crew
        if not callable(getattr(crew, "kickoff", None)):
            raise TypeError(
                "configured CrewAI source must return an object with a callable "
                "kickoff method"
            )
        return crew

    def run(self, prompt: str, **kwargs: Any) -> AgentResponse:
        crew = self._crew_for_run()
        kickoff_inputs = dict(self.inputs)
        nested_inputs = kwargs.pop("inputs", None)
        if nested_inputs is not None:
            if not isinstance(nested_inputs, Mapping):
                raise TypeError("run(inputs=...) must be a mapping")
            kickoff_inputs.update(nested_inputs)
        kickoff_inputs.update(kwargs)
        kickoff_inputs[self.input_key] = prompt

        started = time.perf_counter()
        with _capture_steps(crew) as (steps, observed_tools):
            output = _complete_stream(crew.kickoff(inputs=kickoff_inputs))
        latency_ms = (time.perf_counter() - started) * 1000.0

        tasks_output = _task_outputs(output, crew)
        for task_output in tasks_output:
            observed_tools.extend(_tools_from_messages(_value(task_output, "messages")))
        tool_calls = list(dict.fromkeys(observed_tools))

        usage = _value(output, "token_usage")
        if usage is None:
            usage = _value(crew, "usage_metrics")
        prompt_tokens = _as_int(_value(usage, "prompt_tokens", "input_tokens"))
        completion_tokens = _as_int(
            _value(usage, "completion_tokens", "output_tokens")
        )
        total_tokens = _as_int(_value(usage, "total_tokens"))
        cost_usd = _as_float(_value(usage, "cost_usd", "total_cost"))

        raw = {
            "crew_output": _plain(output),
            "tasks_output": [_plain(item) for item in tasks_output],
            "token_usage": _plain(usage),
            "inputs": _plain(kickoff_inputs),
            "steps": steps,
        }
        return AgentResponse(
            output=_final_answer(output),
            tool_calls=tool_calls,
            nodes_fired=_trajectory(tasks_output, crew),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            raw=raw,
        )


__all__ = ["CrewAIAdapter"]
