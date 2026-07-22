"""OpenAI Agents SDK adapter for AgentEval.

The ``openai-agents`` package is optional and imported lazily. The adapter uses
the documented ``Runner``/``RunResult`` surfaces while retaining duck-typed
runner injection for deterministic tests and custom runner configurations.
"""

from __future__ import annotations

import asyncio
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
    # Imported Agent instances are not normally callable, while factory
    # functions and partials are. Preserve instances and invoke factories.
    if callable(target) and value(target, "name") is None:
        return target()
    return target


def _output_text(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output.strip()
    serialized = plain(output)
    if isinstance(serialized, (dict, list)):
        return json.dumps(serialized, ensure_ascii=False, sort_keys=True)
    return str(serialized).strip()


def _run_items(result: Any) -> list[Any]:
    items = value(result, "new_items")
    if items is None:
        return []
    if isinstance(items, Sequence) and not isinstance(items, (str, bytes)):
        return list(items)
    raise TypeError("OpenAI Agents result.new_items must be a sequence")


def _item_type(item: Any) -> str:
    item_type = value(item, "type")
    if item_type is not None:
        return str(item_type).lower()
    name = type(item).__name__
    return "".join(f"_{character.lower()}" if character.isupper() else character for character in name).lstrip("_")


def _tool_name(item: Any) -> str | None:
    item_type = _item_type(item)
    if "handoff" in item_type or "output" in item_type or "message" in item_type:
        return None

    raw = value(item, "raw_item")
    function = value(raw, "function")
    candidate = value(item, "qualified_name", "tool_name", "name")
    candidate = candidate or value(function, "name")
    candidate = candidate or value(raw, "name", "tool_name")
    normalized = normalized_name(candidate)
    if normalized:
        return normalized

    raw_type = str(value(raw, "type") or "").lower()
    if raw_type.endswith("_call") and raw_type != "function_call":
        return raw_type.removesuffix("_call")
    return None


def _agent_name(agent: Any) -> str | None:
    return normalized_name(value(agent, "name", "role"))


def _trajectory(items: list[Any], result: Any) -> list[str]:
    nodes: list[str] = []

    def add(agent: Any) -> None:
        name = _agent_name(agent)
        node = f"agent:{name}" if name else None
        if node and (not nodes or nodes[-1] != node):
            nodes.append(node)

    for item in items:
        if "handoff_output" in _item_type(item):
            add(value(item, "source_agent"))
            add(value(item, "target_agent"))
        else:
            add(value(item, "agent"))
    if not nodes:
        add(value(result, "last_agent"))
    return nodes


def _runner_usage(result: Any) -> Any:
    context_wrapper = value(result, "context_wrapper")
    return value(context_wrapper, "usage") or value(result, "usage", "token_usage")


def _running_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


class OpenAIAgentsAdapter(AgentAdapter):
    """Run an OpenAI Agents SDK agent and normalize its result.

    Supply an SDK ``Agent`` instance, a factory, or an import path. ``runner``
    defaults to ``agents.Runner`` and is injectable for custom runner classes.
    Token cost is captured when exposed by a provider; optional per-million
    rates provide deterministic cost calculation otherwise.
    """

    def __init__(
        self,
        agent: Any | None = None,
        *,
        agent_factory: Callable[[], Any] | None = None,
        agent_import: str | None = None,
        repo_path: str | Path | None = None,
        runner: Any | None = None,
        run_options: Mapping[str, Any] | None = None,
        input_cost_per_million: float | None = None,
        output_cost_per_million: float | None = None,
    ) -> None:
        sources = sum(item is not None for item in (agent, agent_factory, agent_import))
        if sources != 1:
            raise ValueError("provide exactly one of agent, agent_factory, or agent_import")
        if agent_factory is not None and not callable(agent_factory):
            raise TypeError("agent_factory must be callable")
        validate_import_path(agent_import, "agent_import")
        if runner is not None and not (
            callable(getattr(runner, "run_sync", None))
            or callable(getattr(runner, "run", None))
        ):
            raise TypeError("runner must provide a callable run_sync or run method")
        if run_options is not None and not isinstance(run_options, Mapping):
            raise TypeError("run_options must be a mapping")

        self.agent = agent
        self.agent_factory = agent_factory
        self.agent_import = agent_import
        self.repo_path = Path(repo_path).expanduser().resolve() if repo_path else None
        self.runner = runner
        self.run_options = dict(run_options or {})
        self.input_cost_per_million = validate_rate(
            input_cost_per_million, "input_cost_per_million"
        )
        self.output_cost_per_million = validate_rate(
            output_cost_per_million, "output_cost_per_million"
        )

    def _agent_for_run(self) -> Any:
        if self.agent_import is not None:
            agent = _materialize(import_target(self.agent_import, self.repo_path))
        elif self.agent_factory is not None:
            agent = self.agent_factory()
        else:
            agent = self.agent
        if agent is None:
            raise TypeError("configured OpenAI Agents source returned None")
        return agent

    def _runner(self) -> Any:
        if self.runner is not None:
            return self.runner
        try:
            from agents import Runner
        except ImportError as exc:
            raise ImportError(
                "OpenAI Agents SDK is not installed; install "
                "'nishanttyagi-agenteval[openai-agents]'"
            ) from exc
        return Runner

    def _invoke(self, agent: Any, prompt: str, options: dict[str, Any]) -> Any:
        runner = self._runner()
        async_run = getattr(runner, "run", None)
        sync_run = getattr(runner, "run_sync", None)

        # Runner.run_sync delegates to asyncio.run and explicitly rejects an
        # active event loop. Use Runner.run plus the safe bridge in that case.
        if _running_event_loop() and callable(async_run):
            return resolve_awaitable(async_run(agent, prompt, **options))
        if callable(sync_run):
            return sync_run(agent, prompt, **options)
        if callable(async_run):
            return resolve_awaitable(async_run(agent, prompt, **options))
        raise TypeError("runner must provide a callable run_sync or run method")

    def run(self, prompt: str, **kwargs: Any) -> AgentResponse:
        agent = self._agent_for_run()
        options = dict(self.run_options)
        nested_options = kwargs.pop("run_options", None)
        if nested_options is not None:
            if not isinstance(nested_options, Mapping):
                raise TypeError("run(run_options=...) must be a mapping")
            options.update(nested_options)
        options.update(kwargs)

        started = time.perf_counter()
        result = self._invoke(agent, prompt, options)
        latency_ms = (time.perf_counter() - started) * 1000.0

        items = _run_items(result)
        usage = _runner_usage(result)
        prompt_tokens = as_int(value(usage, "input_tokens", "prompt_tokens"))
        completion_tokens = as_int(value(usage, "output_tokens", "completion_tokens"))
        cost_usd = usage_cost(
            usage,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            input_cost_per_million=self.input_cost_per_million,
            output_cost_per_million=self.output_cost_per_million,
        )
        tools = [name for item in items if (name := _tool_name(item))]
        raw_responses = value(result, "raw_responses") or []
        interruptions = value(result, "interruptions") or []

        return AgentResponse(
            output=_output_text(value(result, "final_output")),
            tool_calls=unique(tools),
            nodes_fired=_trajectory(items, result),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=as_int(value(usage, "total_tokens")),
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            raw={
                "final_output": plain(value(result, "final_output")),
                "new_items": [plain(item) for item in items],
                "raw_responses": plain(raw_responses),
                "usage": plain(usage),
                "last_agent": _agent_name(value(result, "last_agent")),
                "interruptions": plain(interruptions),
                "input": prompt,
                "run_options": plain(options),
            },
        )


__all__ = ["OpenAIAgentsAdapter"]

