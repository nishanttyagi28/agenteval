"""Validated loading and runtime resolution for the agent registry."""

from __future__ import annotations

import importlib
import os
import re
from pathlib import Path
from typing import Any

import yaml

from agenteval.adapters.base import AgentAdapter
from agenteval.core.config import AgentDependencyNotFound
from agenteval.core.schema import AgentConfig, GateConfig, RepositoryConfig

REGISTRY_VERSION = 1
DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parents[1] / "agents.yaml"

_AGENT_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_ENV_VAR_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_ADAPTER_RE = re.compile(
    r"^(?P<module>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*):(?P<class>[A-Za-z_]\w*)$"
)


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"Duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _safe_artifact_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must not be absolute or contain '..': {value!r}")
    return path


def _rate(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number between 0 and 1") from exc
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1")
    return result


def load_adapter_class(import_path: str) -> type[AgentAdapter]:
    """Import an adapter class and verify the AgentAdapter contract."""
    match = _ADAPTER_RE.fullmatch(import_path or "")
    if match is None:
        raise ValueError(
            f"adapter must use 'module.path:ClassName' format, got {import_path!r}"
        )
    try:
        module = importlib.import_module(match.group("module"))
        adapter_class = getattr(module, match.group("class"))
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Cannot import adapter {import_path!r}: {exc}") from exc
    if not isinstance(adapter_class, type) or not issubclass(adapter_class, AgentAdapter):
        raise ValueError(f"Configured adapter {import_path!r} is not an AgentAdapter subclass")
    return adapter_class


def _parse_agent(name: str, raw: Any) -> AgentConfig:
    if not _AGENT_NAME_RE.fullmatch(name):
        raise ValueError(f"Invalid agent name {name!r}; use lowercase letters, digits and underscores")
    data = _mapping(raw, f"agents.{name}")
    adapter = data.get("adapter")
    if not isinstance(adapter, str):
        raise ValueError(f"agents.{name}.adapter must be a string")
    load_adapter_class(adapter)

    repo = _mapping(data.get("repository"), f"agents.{name}.repository")
    env_var = repo.get("env_var")
    if not isinstance(env_var, str) or not _ENV_VAR_RE.fullmatch(env_var):
        raise ValueError(f"agents.{name}.repository.env_var is not a valid environment variable")
    default_path = repo.get("default_path")
    if default_path is not None and (not isinstance(default_path, str) or not default_path.strip()):
        raise ValueError(f"agents.{name}.repository.default_path must be a non-empty string")
    required_raw = repo.get("required_paths") or []
    if not isinstance(required_raw, list):
        raise ValueError(f"agents.{name}.repository.required_paths must be a list")
    required_paths = tuple(
        str(_safe_artifact_path(item, f"agents.{name}.repository.required_paths"))
        for item in required_raw
    )
    ci_repository = repo.get("ci_repository")
    ci_checkout_path = repo.get("ci_checkout_path")
    for value, label in (
        (ci_repository, "ci_repository"),
        (ci_checkout_path, "ci_checkout_path"),
    ):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"agents.{name}.repository.{label} must be a non-empty string")
    if ci_checkout_path is not None:
        _safe_artifact_path(ci_checkout_path, f"agents.{name}.repository.ci_checkout_path")

    gates_raw = _mapping(data.get("gates") or {}, f"agents.{name}.gates")
    gates = GateConfig(
        max_correctness_drop=_rate(
            gates_raw.get("max_correctness_drop", 0.05),
            f"agents.{name}.gates.max_correctness_drop",
        ),
        max_hallucination_rate=_rate(
            gates_raw.get("max_hallucination_rate", 0.10),
            f"agents.{name}.gates.max_hallucination_rate",
        ),
        min_tool_accuracy=_rate(
            gates_raw.get("min_tool_accuracy", 0.90),
            f"agents.{name}.gates.min_tool_accuracy",
        ),
        fail_on_evaluator_error=bool(gates_raw.get("fail_on_evaluator_error", True)),
        fail_on_agent_error=bool(gates_raw.get("fail_on_agent_error", True)),
    )
    options = data.get("adapter_options") or {}
    if not isinstance(options, dict):
        raise ValueError(f"agents.{name}.adapter_options must be a mapping")
    smoke_raw = data.get("smoke_case_ids") or []
    if not isinstance(smoke_raw, list) or not all(
        isinstance(case_id, str) and case_id.strip() for case_id in smoke_raw
    ):
        raise ValueError(f"agents.{name}.smoke_case_ids must be a list of case ids")

    return AgentConfig(
        name=name,
        display_name=str(data.get("display_name") or name),
        adapter=adapter,
        repository=RepositoryConfig(
            env_var=env_var,
            default_path=default_path,
            required_paths=required_paths,
            ci_repository=ci_repository,
            ci_checkout_path=ci_checkout_path,
        ),
        golden_suite=_safe_artifact_path(data.get("golden_suite"), f"agents.{name}.golden_suite"),
        baseline=_safe_artifact_path(data.get("baseline"), f"agents.{name}.baseline"),
        runs_dir=_safe_artifact_path(data.get("runs_dir"), f"agents.{name}.runs_dir"),
        enabled=bool(data.get("enabled", True)),
        adapter_options=dict(options),
        gates=gates,
        smoke_case_ids=tuple(smoke_raw),
    )


def load_agent_registry(path: str | Path = DEFAULT_REGISTRY_PATH) -> dict[str, AgentConfig]:
    """Load and fully validate ``agents.yaml`` without requiring sibling repos."""
    registry_path = Path(path)
    try:
        raw = yaml.load(registry_path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid agent registry YAML in {registry_path}: {exc}") from exc
    root = _mapping(raw, "agent registry")
    if root.get("version") != REGISTRY_VERSION:
        raise ValueError(
            f"Unsupported agent registry version {root.get('version')!r}; expected {REGISTRY_VERSION}"
        )
    agents_raw = _mapping(root.get("agents"), "agents")
    if not agents_raw:
        raise ValueError("Agent registry must contain at least one agent")
    return {name: _parse_agent(name, item) for name, item in agents_raw.items()}


def resolve_agent_repository(
    config: AgentConfig,
    *,
    explicit: str | Path | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> Path:
    """Resolve a configured sibling repository or raise a typed error."""
    registry_dir = Path(registry_path).resolve().parent
    candidates: list[Path] = []
    for raw in (explicit, os.getenv(config.repository.env_var), config.repository.default_path):
        if raw is None or not str(raw).strip():
            continue
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = registry_dir / path
        resolved = path.resolve()
        if resolved not in candidates:
            candidates.append(resolved)
    for candidate in candidates:
        if candidate.is_dir() and all(
            (candidate / required).exists() for required in config.repository.required_paths
        ):
            return candidate
    checked = "\n".join(f"  - {candidate}" for candidate in candidates) or "  - (none)"
    raise AgentDependencyNotFound(
        f"{config.display_name} dependency not found. Set {config.repository.env_var} "
        f"or provide an explicit repository path.\nChecked:\n{checked}"
    )
