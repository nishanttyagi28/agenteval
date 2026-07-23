"""LangGraph adapter for AgentEval.

LangGraph is an optional dependency and is never imported by this module —
the adapter drives any object exposing the documented compiled-graph
``stream(input, config=..., stream_mode="updates")`` contract via duck
typing, so tests can inject lightweight fakes without installing the real
package.

LangGraph has no single "final answer" field the way a chat completion does:
the graph mutates a shared state object across node executions. This adapter
reconstructs, from the ``"updates"`` stream, the ordered list of node
executions (``nodes_fired`` — including repeats, so a node revisited in a
cycle reads as a retry), a best-effort merged final state (later deltas win
per key), invoked tool names, and token usage summed off any LangChain
message objects seen in the deltas. ``output_key`` (or a documented fallback
chain) tells the adapter which state key holds the answer.
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
    unique,
    usage_cost,
    validate_import_path,
    validate_rate,
    value,
)
from agenteval.adapters.base import AgentAdapter, AgentResponse

_DEFAULT_OUTPUT_KEYS: tuple[str, ...] = ("output", "answer", "response", "result")
_AI_ROLES = {"ai", "assistant", "aimessage", "aimessagechunk"}
_TOOL_ROLES = {"tool", "toolmessage"}


def _materialize(target: Any) -> Any:
    if isinstance(target, type):
        return target()
    if callable(getattr(target, "stream", None)) or callable(getattr(target, "invoke", None)):
        return target
    if callable(target):
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


def _default_input(input_key: str, prompt: str) -> dict[str, Any]:
    if input_key == "messages":
        return {"messages": [{"role": "user", "content": prompt}]}
    return {input_key: prompt}


def _split_stream_item(item: Any) -> tuple[tuple[str, ...], Mapping[str, Any]]:
    """Return ``(namespace, update)`` for one ``stream_mode="updates"`` item.

    Plain-graph streaming yields ``{node_name: state_delta}``. Subgraph
    streaming (``subgraphs=True``) yields ``(namespace_tuple, {node_name:
    state_delta})``; both shapes are accepted so the adapter tolerates either
    configuration without extra caller-side branching.
    """
    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], Mapping):
        namespace, update = item
        if isinstance(namespace, (list, tuple)):
            return tuple(str(part) for part in namespace), update
        return (str(namespace),), update
    if isinstance(item, Mapping):
        return (), item
    return (), {}


def _iter_messages(node_output: Any) -> list[Any]:
    """Pull message-shaped objects out of one node's state delta, if any."""
    candidates: Any = None
    if isinstance(node_output, Mapping):
        candidates = node_output.get("messages")
    elif isinstance(node_output, Sequence) and not isinstance(node_output, (str, bytes)):
        candidates = node_output
    if candidates is None:
        return []
    if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
        return list(candidates)
    return [candidates]


def _message_role(message: Any) -> str | None:
    role = value(message, "type", "role")
    return str(role).lower() if role is not None else None


def _tool_call_names(message: Any) -> list[str]:
    names: list[str] = []
    calls = value(message, "tool_calls")
    if isinstance(calls, Sequence) and not isinstance(calls, (str, bytes)):
        for call in calls:
            function = value(call, "function")
            name = normalized_name(value(call, "name") or value(function, "name"))
            if name:
                names.append(name)
    if _message_role(message) in _TOOL_ROLES:
        name = normalized_name(value(message, "name", "tool_name"))
        if name:
            names.append(name)
    return names


def _message_usage(message: Any) -> tuple[int | None, int | None, float | None]:
    usage = value(message, "usage_metadata")
    if usage is None:
        usage = value(value(message, "response_metadata"), "token_usage")
    if usage is None:
        return None, None, None
    prompt = as_int(value(usage, "input_tokens", "prompt_tokens"))
    completion = as_int(value(usage, "output_tokens", "completion_tokens"))
    cost = as_float(value(usage, "cost_usd", "total_cost", "cost"))
    return prompt, completion, cost


class LangGraphAdapter(AgentAdapter):
    """Run a compiled LangGraph graph and normalize its update stream.

    Supply a compiled graph instance, a factory, or an import path via
    ``graph``, ``graph_factory``, or ``graph_import`` (exactly one). A direct
    ``graph`` is reused across cases; ``graph_factory``/``graph_import``
    build a fresh graph per case, which is generally preferable when the
    graph carries a checkpointer or other per-run state.
    """

    def __init__(
        self,
        graph: Any | None = None,
        *,
        graph_factory: Callable[[], Any] | None = None,
        graph_import: str | None = None,
        repo_path: str | Path | None = None,
        input_key: str = "messages",
        input_builder: Callable[[str], Any] | None = None,
        output_key: str | None = None,
        config: Mapping[str, Any] | None = None,
        input_cost_per_million: float | None = None,
        output_cost_per_million: float | None = None,
    ) -> None:
        sources = sum(item is not None for item in (graph, graph_factory, graph_import))
        if sources != 1:
            raise ValueError("provide exactly one of graph, graph_factory, or graph_import")
        if graph is not None and not callable(getattr(graph, "stream", None)):
            raise TypeError("graph must provide a callable stream method")
        if graph_factory is not None and not callable(graph_factory):
            raise TypeError("graph_factory must be callable")
        validate_import_path(graph_import, "graph_import")
        if not isinstance(input_key, str) or not input_key.strip():
            raise ValueError("input_key must be a non-empty string")
        if input_builder is not None and not callable(input_builder):
            raise TypeError("input_builder must be callable")
        if output_key is not None and (not isinstance(output_key, str) or not output_key.strip()):
            raise ValueError("output_key must be a non-empty string or None")
        if config is not None and not isinstance(config, Mapping):
            raise TypeError("config must be a mapping")

        self.graph = graph
        self.graph_factory = graph_factory
        self.graph_import = graph_import
        self.repo_path = Path(repo_path).expanduser().resolve() if repo_path else None
        self.input_key = input_key.strip()
        self.input_builder = input_builder
        self.output_key = output_key.strip() if output_key else None
        self.config = dict(config or {})
        self.input_cost_per_million = validate_rate(
            input_cost_per_million, "input_cost_per_million"
        )
        self.output_cost_per_million = validate_rate(
            output_cost_per_million, "output_cost_per_million"
        )

    def _graph_for_run(self) -> Any:
        if self.graph_import is not None:
            graph = _materialize(import_target(self.graph_import, self.repo_path))
        elif self.graph_factory is not None:
            graph = self.graph_factory()
        else:
            graph = self.graph
        if not callable(getattr(graph, "stream", None)):
            raise TypeError(
                "configured LangGraph source must return an object with a callable stream method"
            )
        return graph

    def _build_input(self, prompt: str) -> Any:
        if self.input_builder is not None:
            return self.input_builder(prompt)
        return _default_input(self.input_key, prompt)

    def _resolve_output(self, state: dict[str, Any]) -> str:
        if self.output_key is not None:
            return _output_text(state.get(self.output_key))
        for key in _DEFAULT_OUTPUT_KEYS:
            if key in state:
                return _output_text(state[key])
        messages = state.get("messages")
        if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
            for message in reversed(list(messages)):
                role = _message_role(message)
                if role is None or role in _AI_ROLES:
                    content = value(message, "content")
                    if content is not None:
                        return _output_text(content)
        return _output_text(state) if state else ""

    def run(self, prompt: str, **kwargs: Any) -> AgentResponse:
        graph = self._graph_for_run()
        inputs = self._build_input(prompt)
        run_config = dict(self.config)
        nested_config = kwargs.pop("config", None)
        if nested_config is not None:
            if not isinstance(nested_config, Mapping):
                raise TypeError("run(config=...) must be a mapping")
            run_config.update(nested_config)
        run_config.update(kwargs)

        started = time.perf_counter()
        events = list(graph.stream(inputs, config=run_config or None, stream_mode="updates"))
        latency_ms = (time.perf_counter() - started) * 1000.0

        nodes_fired: list[str] = []
        node_visit_counts: dict[str, int] = {}
        merged_state: dict[str, Any] = {}
        tool_names: list[str] = []
        prompt_tokens_total = 0
        completion_tokens_total = 0
        cost_total = 0.0
        saw_prompt = False
        saw_completion = False
        saw_cost = False
        raw_events: list[Any] = []

        for item in events:
            raw_events.append(plain(item))
            namespace, update = _split_stream_item(item)
            if not isinstance(update, Mapping):
                continue
            for node_name, delta in update.items():
                node_key = "/".join((*namespace, str(node_name))) if namespace else str(node_name)
                nodes_fired.append(node_key)
                node_visit_counts[node_key] = node_visit_counts.get(node_key, 0) + 1
                if isinstance(delta, Mapping):
                    merged_state.update(delta)
                for message in _iter_messages(delta):
                    tool_names.extend(_tool_call_names(message))
                    prompt_t, completion_t, cost_t = _message_usage(message)
                    if prompt_t is not None:
                        prompt_tokens_total += prompt_t
                        saw_prompt = True
                    if completion_t is not None:
                        completion_tokens_total += completion_t
                        saw_completion = True
                    if cost_t is not None:
                        cost_total += cost_t
                        saw_cost = True

        prompt_tokens = prompt_tokens_total if saw_prompt else None
        completion_tokens = completion_tokens_total if saw_completion else None
        usage: dict[str, Any] = {}
        if saw_prompt:
            usage["prompt_tokens"] = prompt_tokens
        if saw_completion:
            usage["completion_tokens"] = completion_tokens
        if saw_cost:
            usage["cost_usd"] = cost_total
        cost_usd = usage_cost(
            usage or None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            input_cost_per_million=self.input_cost_per_million,
            output_cost_per_million=self.output_cost_per_million,
        )
        retries = {name: count for name, count in node_visit_counts.items() if count > 1}

        return AgentResponse(
            output=self._resolve_output(merged_state),
            tool_calls=unique(tool_names),
            nodes_fired=nodes_fired,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            raw={
                "events": raw_events,
                "node_visit_counts": node_visit_counts,
                "retries": retries,
                "execution_path": list(nodes_fired),
                "final_state": plain(merged_state),
                "input": plain(inputs),
                "config_keys": sorted(str(key) for key in run_config),
            },
        )


__all__ = ["LangGraphAdapter"]
