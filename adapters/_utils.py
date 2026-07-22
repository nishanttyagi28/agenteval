"""Shared normalization helpers for optional third-party agent adapters."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import sys
from collections.abc import Awaitable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def value(source: Any, *names: str) -> Any:
    """Return the first non-``None`` mapping value or object attribute."""
    for name in names:
        if isinstance(source, Mapping) and name in source:
            candidate = source[name]
        else:
            try:
                candidate = getattr(source, name)
            except (AttributeError, TypeError):
                continue
        if candidate is not None:
            return candidate
    return None


def plain(item: Any) -> Any:
    """Convert SDK and Pydantic values into JSON-safe evidence."""
    if item is None or isinstance(item, (str, int, float, bool)):
        return item
    if isinstance(item, Mapping):
        return {str(key): plain(entry) for key, entry in item.items()}
    if isinstance(item, (list, tuple, set)):
        return [plain(entry) for entry in item]
    if is_dataclass(item) and not isinstance(item, type):
        return plain(asdict(item))
    model_dump = getattr(item, "model_dump", None)
    if callable(model_dump):
        try:
            return plain(model_dump(mode="json"))
        except Exception:  # noqa: BLE001 - evidence collection must be best effort
            try:
                return plain(model_dump())
            except Exception:  # noqa: BLE001
                pass
    to_dict = getattr(item, "to_dict", None)
    if callable(to_dict):
        try:
            return plain(to_dict())
        except Exception:  # noqa: BLE001
            pass
    try:
        return str(item)
    except Exception:  # noqa: BLE001
        return f"<{type(item).__name__}>"


def as_int(item: Any) -> int | None:
    if item is None or isinstance(item, bool):
        return None
    try:
        result = int(item)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def as_float(item: Any) -> float | None:
    if item is None or isinstance(item, bool):
        return None
    try:
        result = float(item)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def normalized_name(item: Any) -> str | None:
    if item is None:
        return None
    if not isinstance(item, str):
        item = value(item, "name", "role") or item
    result = str(item).strip()
    return result or None


def unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def import_target(import_path: str, repo_path: Path | None) -> Any:
    """Import ``module.path:object`` after exposing a sibling repo and its src dir."""
    module_name, symbol_name = import_path.split(":", 1)
    if repo_path is not None:
        for candidate in (repo_path, repo_path / "src"):
            candidate_text = str(candidate)
            if candidate.is_dir() and candidate_text not in sys.path:
                sys.path.insert(0, candidate_text)
    try:
        return getattr(importlib.import_module(module_name), symbol_name)
    except (ImportError, AttributeError) as exc:
        raise ImportError(f"cannot import entrypoint {import_path!r}: {exc}") from exc


async def _await_result(awaitable: Awaitable[Any]) -> Any:
    return await awaitable


def resolve_awaitable(result: Any) -> Any:
    """Resolve an async SDK result from synchronous code, even inside an event loop."""
    if not inspect.isawaitable(result):
        return result
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_await_result(result))

    # ``asyncio.run`` cannot be nested. A short-lived worker preserves the
    # synchronous AgentAdapter contract for notebook and async-web callers.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="agenteval-adapter") as pool:
        return pool.submit(lambda: asyncio.run(_await_result(result))).result()


def validate_import_path(import_path: str | None, label: str) -> None:
    if import_path is None:
        return
    if not isinstance(import_path, str) or import_path.count(":") != 1:
        raise ValueError(f"{label} must use 'module.path:Name' format")
    module_name, symbol_name = import_path.split(":", 1)
    if not module_name.strip() or not symbol_name.strip():
        raise ValueError(f"{label} must use 'module.path:Name' format")


def validate_rate(rate: float | None, label: str) -> float | None:
    if rate is None:
        return None
    try:
        normalized = float(rate)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be a non-negative number or None") from exc
    if normalized < 0:
        raise ValueError(f"{label} must be non-negative")
    return normalized


def usage_cost(
    usage: Any,
    *,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    input_cost_per_million: float | None,
    output_cost_per_million: float | None,
) -> float | None:
    """Read provider cost or calculate it from explicitly configured token rates."""
    exposed = as_float(value(usage, "cost_usd", "total_cost", "cost"))
    if exposed is not None:
        return exposed
    if input_cost_per_million is None and output_cost_per_million is None:
        return None
    prompt_cost = (prompt_tokens or 0) * (input_cost_per_million or 0) / 1_000_000
    completion_cost = (
        (completion_tokens or 0) * (output_cost_per_million or 0) / 1_000_000
    )
    return prompt_cost + completion_cost

