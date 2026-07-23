"""Lazy, deterministic discovery and loading for evaluator entry points."""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, replace
from importlib import metadata
from typing import Any, Iterable

from agenteval import __version__
from agenteval.evaluators import EvaluationContext, EvaluationResult, Evaluator

ENTRY_POINT_GROUP = "agenteval.evaluators"
BUILTIN_EVALUATORS: tuple[str, ...] = (
    "contains",
    "exact",
    "llm_judge",
    "numeric",
    "numeric_table",
)

_NAME_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
_TARGET_RE = re.compile(
    r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:"
    r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$"
)


class EvaluatorPluginError(RuntimeError):
    """Base class for actionable evaluator-plugin failures."""


class EvaluatorNotFoundError(EvaluatorPluginError):
    """Raised when no evaluator is registered under a requested name."""


class EvaluatorDiscoveryError(EvaluatorPluginError):
    """Raised when installed entry-point metadata cannot be enumerated."""


class DuplicateEvaluatorError(EvaluatorPluginError):
    """Raised when an evaluator name does not resolve uniquely."""


class EvaluatorLoadError(EvaluatorPluginError):
    """Raised when an evaluator entry point cannot be imported."""


class EvaluatorValidationError(EvaluatorPluginError):
    """Raised when a loaded evaluator does not implement the contract."""


class EvaluatorExecutionError(EvaluatorPluginError):
    """Raised when evaluator execution or result validation fails."""


class EvaluatorDependencyError(EvaluatorLoadError):
    """Raised when a plugin's optional dependency is unavailable."""


@dataclass(frozen=True)
class EvaluatorInfo:
    """Metadata-only description of a built-in or third-party evaluator."""

    name: str
    source: str
    package: str
    version: str
    target: str | None
    status: str
    diagnostic: str | None = None


@dataclass(frozen=True)
class _Candidate:
    info: EvaluatorInfo
    entry_point: Any | None = None


def _distribution_metadata(entry_point: Any) -> tuple[str, str]:
    distribution = getattr(entry_point, "dist", None)
    if distribution is None:
        return "(unknown)", "(unknown)"
    package = None
    dist_metadata = getattr(distribution, "metadata", None)
    if dist_metadata is not None:
        try:
            package = dist_metadata.get("Name")
        except AttributeError:
            package = None
    package = str(package or getattr(distribution, "name", None) or "(unknown)")
    version = str(getattr(distribution, "version", None) or "(unknown)")
    return package, version


def _installed_entry_points() -> list[Any]:
    try:
        discovered = metadata.entry_points()
    except Exception as exc:  # noqa: BLE001 - normalize metadata backend failures
        raise EvaluatorDiscoveryError(
            f"Cannot enumerate {ENTRY_POINT_GROUP!r} entry points: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    select = getattr(discovered, "select", None)
    if callable(select):
        return list(select(group=ENTRY_POINT_GROUP))
    if isinstance(discovered, dict):  # pragma: no cover - Python <3.10 compatibility
        return list(discovered.get(ENTRY_POINT_GROUP, ()))
    return [
        item
        for item in discovered
        if getattr(item, "group", ENTRY_POINT_GROUP) == ENTRY_POINT_GROUP
    ]


def _candidate(entry_point: Any) -> _Candidate:
    name = str(getattr(entry_point, "name", ""))
    target = str(getattr(entry_point, "value", ""))
    package, version = _distribution_metadata(entry_point)
    diagnostic = None
    status = "discovered"
    if not _NAME_RE.fullmatch(name):
        status = "malformed"
        diagnostic = (
            f"invalid evaluator name {name!r}; expected lowercase letters, digits, "
            "'.', '_' or '-', starting with a letter"
        )
    elif not _TARGET_RE.fullmatch(target):
        status = "malformed"
        diagnostic = (
            f"malformed entry-point target {target!r}; expected "
            "'module.path:callable_name'"
        )
    elif name in BUILTIN_EVALUATORS:
        status = "duplicate"
        diagnostic = f"evaluator name {name!r} is reserved by a built-in evaluator"
    return _Candidate(
        info=EvaluatorInfo(
            name=name,
            source="third-party",
            package=package,
            version=version,
            target=target,
            status=status,
            diagnostic=diagnostic,
        ),
        entry_point=entry_point,
    )


def _sort_key(candidate: _Candidate) -> tuple[str, str, str, str]:
    info = candidate.info
    return (
        info.name,
        info.package.casefold(),
        info.version,
        info.target or "",
    )


def _discover_candidates(entry_points: Iterable[Any] | None = None) -> list[_Candidate]:
    items = _installed_entry_points() if entry_points is None else list(entry_points)
    candidates = sorted((_candidate(item) for item in items), key=_sort_key)
    counts: dict[str, int] = {}
    for candidate in candidates:
        if (
            _NAME_RE.fullmatch(candidate.info.name)
            and candidate.info.name not in BUILTIN_EVALUATORS
        ):
            counts[candidate.info.name] = counts.get(candidate.info.name, 0) + 1
    output: list[_Candidate] = []
    for candidate in candidates:
        if candidate.info.status == "discovered" and counts[candidate.info.name] > 1:
            candidate = replace(
                candidate,
                info=replace(
                    candidate.info,
                    status="duplicate",
                    diagnostic=(
                        f"{counts[candidate.info.name]} third-party entry points "
                        f"register evaluator name {candidate.info.name!r}"
                    ),
                ),
            )
        output.append(candidate)
    return output


def discover_evaluators(entry_points: Iterable[Any] | None = None) -> tuple[EvaluatorInfo, ...]:
    """Return deterministic metadata without importing third-party code."""
    builtins = [
        EvaluatorInfo(
            name=name,
            source="built-in",
            package="nishanttyagi-agenteval",
            version=__version__,
            target=None,
            status="available",
        )
        for name in BUILTIN_EVALUATORS
    ]
    third_party = [item.info for item in _discover_candidates(entry_points)]
    return tuple(sorted((*builtins, *third_party), key=lambda item: (item.name, item.source)))


def evaluator_info(
    name: str,
    entry_points: Iterable[Any] | None = None,
) -> tuple[EvaluatorInfo, ...]:
    """Return every descriptor matching ``name`` without loading code."""
    return tuple(
        info for info in discover_evaluators(entry_points) if info.name == name
    )


def _selected_candidate(
    name: str,
    entry_points: Iterable[Any] | None,
) -> _Candidate:
    if name in BUILTIN_EVALUATORS:
        raise EvaluatorValidationError(
            f"{name!r} is a built-in evaluator and does not load through a third-party entry point"
        )
    matches = [
        candidate
        for candidate in _discover_candidates(entry_points)
        if candidate.info.name == name
    ]
    if not matches:
        raise EvaluatorNotFoundError(
            f"Unknown evaluator {name!r}. Run 'agenteval plugins list' to see available names."
        )
    if len(matches) != 1 or matches[0].info.status == "duplicate":
        details = ", ".join(
            f"{item.info.package} {item.info.version} ({item.info.target})"
            for item in matches
        )
        raise DuplicateEvaluatorError(
            f"Evaluator name {name!r} is ambiguous: {details}"
        )
    selected = matches[0]
    if selected.info.status != "discovered":
        raise EvaluatorValidationError(
            selected.info.diagnostic
            or f"Evaluator {name!r} has invalid entry-point metadata"
        )
    return selected


def _validate_callable(name: str, plugin: Any) -> Evaluator:
    if not callable(plugin):
        raise EvaluatorValidationError(
            f"Evaluator {name!r} resolved to {type(plugin).__name__}, not a callable"
        )
    try:
        signature = inspect.signature(plugin)
    except (TypeError, ValueError) as exc:
        raise EvaluatorValidationError(
            f"Cannot inspect evaluator {name!r} callable signature: {exc}"
        ) from exc
    try:
        signature.bind(object())
    except TypeError as exc:
        raise EvaluatorValidationError(
            f"Evaluator {name!r} must accept exactly one positional EvaluationContext: {exc}"
        ) from exc
    return plugin


def load_evaluator(
    name: str,
    entry_points: Iterable[Any] | None = None,
) -> Evaluator:
    """Load and structurally validate one explicitly selected evaluator."""
    candidate = _selected_candidate(name, entry_points)
    try:
        plugin = candidate.entry_point.load()
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise EvaluatorDependencyError(
            f"Cannot load evaluator {name!r} from {candidate.info.package}: "
            f"optional dependency/module {missing!r} is unavailable. Install the "
            "plugin package with the dependencies or extras it documents."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - normalize arbitrary import-time failures
        raise EvaluatorLoadError(
            f"Cannot load evaluator {name!r} from {candidate.info.package} "
            f"({candidate.info.target}): {type(exc).__name__}: {exc}"
        ) from exc
    return _validate_callable(name, plugin)


def evaluate(
    name: str,
    context: EvaluationContext,
    entry_points: Iterable[Any] | None = None,
) -> EvaluationResult:
    """Load, invoke, and validate one selected evaluator."""
    plugin = load_evaluator(name, entry_points)
    try:
        result = plugin(context)
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise EvaluatorExecutionError(
            f"Evaluator {name!r} requires unavailable dependency/module {missing!r}"
        ) from exc
    except Exception as exc:  # noqa: BLE001 - plugin failures become evaluator errors
        raise EvaluatorExecutionError(
            f"Evaluator {name!r} raised {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(result, EvaluationResult):
        raise EvaluatorExecutionError(
            f"Evaluator {name!r} returned {type(result).__name__}; expected EvaluationResult"
        )
    if type(result.passed) is not bool:
        raise EvaluatorExecutionError(
            f"Evaluator {name!r} returned non-boolean passed={result.passed!r}"
        )
    if result.reason is not None and not isinstance(result.reason, str):
        raise EvaluatorExecutionError(
            f"Evaluator {name!r} returned a non-string reason"
        )
    return result
