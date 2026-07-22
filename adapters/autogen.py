"""Microsoft AutoGen AgentChat adapter for AgentEval.

AutoGen is an optional dependency. This module targets the documented
``TaskRunner.run(task=...)``/``TaskResult.messages`` contract with duck typing,
so importing AgentEval never imports AutoGen itself.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from agenteval.adapters._utils import (
    as_float,
    as_int,
    import_target,
    normalized_name,
    plain,
    resolve_awaitable,
    unique,
    usage_cost,
    validate_import_path,
    validate_rate,
    value,
)
from agenteval.adapters.base import AgentAdapter, AgentResponse


def _materialize(target: Any) -> Any:
    if isinstance(target, type):
        return target()
    if callable(getattr(target, "run", None)):
        return target
    if callable(target):
        return target()
    return target


def _messages(result: Any) -> list[Any]:
    messages = value(result, "messages")
    if messages is None:
        return []
    if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
        return list(messages)
    raise TypeError("AutoGen result.messages must be a sequence")


def _tool_names(message: Any) -> list[str]:
    candidates: list[Any] = []
    content = value(message, "content")
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        candidates.extend(content)
    calls = value(message, "tool_calls")
    if isinstance(calls, Sequence) and not isinstance(calls, (str, bytes)):
        candidates.extend(calls)
    elif calls is not None:
        candidates.append(calls)

    names: list[str] = []
    for candidate in candidates:
        function = value(candidate, "function")
        name = value(function, "name") if function is not None else None
        name = normalized_name(name or value(candidate, "name", "tool_name", "tool"))
        if name:
            names.append(name)
    return names


def _message_usage(messages: list[Any]) -> tuple[int | None, int | None, float | None]:
    prompt_total = 0
    completion_total = 0
    cost_total = 0.0
    saw_prompt = False
    saw_completion = False
    saw_cost = False
    for message in messages:
        usage = value(message, "models_usage", "model_usage", "usage")
        prompt = as_int(value(usage, "prompt_tokens", "input_tokens"))
        completion = as_int(value(usage, "completion_tokens", "output_tokens"))
        cost = as_float(value(usage, "cost_usd", "total_cost", "cost"))
        if prompt is not None:
            prompt_total += prompt
            saw_prompt = True
        if completion is not None:
            completion_total += completion
            saw_completion = True
        if cost is not None:
            cost_total += cost
            saw_cost = True
    return (
        prompt_total if saw_prompt else None,
        completion_total if saw_completion else None,
        cost_total if saw_cost else None,
    )


def _final_output(result: Any, messages: list[Any]) -> str:
    direct = value(result, "final_output", "output")
    if direct is not None:
        return _output_text(direct)
    for message in reversed(messages):
        content = value(message, "content")
        if isinstance(content, str):
            return content.strip()
    return ""


def _output_text(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output.strip()
    serialized = plain(output)
    if isinstance(serialized, (dict, list)):
        return json.dumps(serialized, ensure_ascii=False, sort_keys=True)
    return str(serialized).strip()


def _trajectory(messages: list[Any]) -> list[str]:
    nodes: list[str] = []
    previous: str | None = None
    for message in messages:
        source = normalized_name(value(message, "source", "sender"))
        if source is None or source.lower() in {"user", "system"}:
            continue
        node = f"agent:{source}"
        if node != previous:
            nodes.append(node)
            previous = node
    return nodes


class AutoGenAdapter(AgentAdapter):
    """Normalize an AutoGen AgentChat agent or team into ``AgentResponse``.

    A direct ``agent`` is reused and therefore keeps AutoGen's documented
    conversation state. ``agent_factory`` and ``agent_import`` create a fresh
    runner for every case, which is generally preferable for isolated evals.
    """

    def __init__(
        self,
        agent: Any | None = None,
        *,
        agent_factory: Callable[[], Any] | None = None,
        agent_import: str | None = None,
        repo_path: str | Path | None = None,
        task_key: str = "task",
        run_options: Mapping[str, Any] | None = None,
        input_cost_per_million: float | None = None,
        output_cost_per_million: float | None = None,
    ) -> None:
        sources = sum(item is not None for item in (agent, agent_factory, agent_import))
        if sources != 1:
            raise ValueError("provide exactly one of agent, agent_factory, or agent_import")
        if agent is not None and not callable(getattr(agent, "run", None)):
            raise TypeError("agent must provide a callable run method")
        if agent_factory is not None and not callable(agent_factory):
            raise TypeError("agent_factory must be callable")
        validate_import_path(agent_import, "agent_import")
        if not isinstance(task_key, str) or not task_key.strip():
            raise ValueError("task_key must be a non-empty string")
        if run_options is not None and not isinstance(run_options, Mapping):
            raise TypeError("run_options must be a mapping")

        self.agent = agent
        self.agent_factory = agent_factory
        self.agent_import = agent_import
        self.repo_path = Path(repo_path).expanduser().resolve() if repo_path else None
        self.task_key = task_key.strip()
        self.run_options = dict(run_options or {})
        self.input_cost_per_million = validate_rate(
            input_cost_per_million, "input_cost_per_million"
        )
        self.output_cost_per_million = validate_rate(
            output_cost_per_million, "output_cost_per_million"
        )

    def _agent_for_run(self) -> Any:
        if self.agent_import is not None:
            runner = _materialize(import_target(self.agent_import, self.repo_path))
        elif self.agent_factory is not None:
            runner = self.agent_factory()
        else:
            runner = self.agent
        if not callable(getattr(runner, "run", None)):
            raise TypeError(
                "configured AutoGen source must return an object with a callable run method"
            )
        return runner

    def run(self, prompt: str, **kwargs: Any) -> AgentResponse:
        runner = self._agent_for_run()
        invocation = dict(self.run_options)
        nested_options = kwargs.pop("run_options", None)
        if nested_options is not None:
            if not isinstance(nested_options, Mapping):
                raise TypeError("run(run_options=...) must be a mapping")
            invocation.update(nested_options)
        invocation.update(kwargs)
        invocation[self.task_key] = prompt

        started = time.perf_counter()
        result = resolve_awaitable(runner.run(**invocation))
        latency_ms = (time.perf_counter() - started) * 1000.0

        messages = _messages(result)
        aggregate_usage = value(result, "usage", "token_usage", "models_usage")
        if aggregate_usage is not None:
            prompt_tokens = as_int(
                value(aggregate_usage, "prompt_tokens", "input_tokens")
            )
            completion_tokens = as_int(
                value(aggregate_usage, "completion_tokens", "output_tokens")
            )
            observed_cost = as_float(
                value(aggregate_usage, "cost_usd", "total_cost", "cost")
            )
        else:
            prompt_tokens, completion_tokens, observed_cost = _message_usage(messages)

        cost_usd = observed_cost
        if cost_usd is None:
            cost_usd = usage_cost(
                aggregate_usage,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                input_cost_per_million=self.input_cost_per_million,
                output_cost_per_million=self.output_cost_per_million,
            )

        tools = [name for message in messages for name in _tool_names(message)]
        return AgentResponse(
            output=_final_output(result, messages),
            tool_calls=unique(tools),
            nodes_fired=_trajectory(messages),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=as_int(value(aggregate_usage, "total_tokens")),
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            raw={
                "result": plain(result),
                "messages": [plain(message) for message in messages],
                "invocation": {
                    "task_key": self.task_key,
                    "prompt": prompt,
                    "option_keys": sorted(
                        str(key) for key in invocation if key != self.task_key
                    ),
                },
            },
        )


__all__ = ["AutoGenAdapter"]
