"""Bundled evaluation-template catalog and safe installer."""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Protocol, Sequence

from agenteval.core._fsutil import atomic_write_text
from agenteval.core.registry import load_agent_registry
from agenteval.core.schema import load_test_cases

TEMPLATE_SCHEMA_VERSION = 1
_NAME_RE = re.compile(r"^[a-z][a-z0-9-]*$")


class TemplateError(ValueError):
    """Base class for template catalog and installation errors."""


class TemplateNotFoundError(TemplateError):
    """Raised when a requested template is not in the catalog."""


class TemplateValidationError(TemplateError):
    """Raised when template metadata or AgentEval files are invalid."""


class TemplateInstallError(TemplateError):
    """Raised when a safe template installation cannot be completed."""


@dataclass(frozen=True)
class TemplateInfo:
    """Validated metadata for one evaluation template."""

    name: str
    title: str
    description: str
    config: str
    cases: str
    case_count: int
    files: tuple[str, ...]
    source: str = "bundled"


class TemplateProvider(Protocol):
    """Internal source interface; only the bundled provider ships in Tier 8."""

    def list_templates(self) -> Sequence[TemplateInfo]:
        """Return metadata for templates supplied by this provider."""
        ...

    def read_files(self, template: TemplateInfo) -> dict[str, bytes]:
        """Return every installable file for ``template``."""
        ...


def _safe_relative(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TemplateValidationError(f"{label} must be a non-empty relative path")
    if "\\" in value:
        raise TemplateValidationError(f"{label} must use forward slashes")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise TemplateValidationError(
            f"{label} must not be absolute or contain '..': {value!r}"
        )
    return path.as_posix()


def _required_string(data: dict[str, object], key: str, source: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TemplateValidationError(
            f"{source}: {key} must be a non-empty string"
        )
    return value.strip()


class BundledTemplateProvider:
    """Read templates packaged below ``agenteval.templates/catalog``."""

    source = "bundled"

    def __init__(self) -> None:
        self._root = resources.files("agenteval.templates").joinpath("catalog")

    def _metadata(self, directory) -> TemplateInfo:
        metadata_path = directory.joinpath("template.json")
        if not metadata_path.is_file():
            raise TemplateValidationError(
                f"bundled template {directory.name!r} is missing template.json"
            )
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise TemplateValidationError(
                f"{directory.name}/template.json is invalid JSON: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise TemplateValidationError(
                f"{directory.name}/template.json must contain a JSON object"
            )
        if data.get("schema_version") != TEMPLATE_SCHEMA_VERSION:
            raise TemplateValidationError(
                f"{directory.name}/template.json has unsupported schema_version "
                f"{data.get('schema_version')!r}; expected {TEMPLATE_SCHEMA_VERSION}"
            )
        name = _required_string(data, "name", directory.name)
        if not _NAME_RE.fullmatch(name) or name != directory.name:
            raise TemplateValidationError(
                f"{directory.name}/template.json: name must match its lowercase "
                "catalog directory and use letters, digits, and hyphens"
            )
        files_raw = data.get("files")
        if not isinstance(files_raw, list) or not files_raw:
            raise TemplateValidationError(
                f"{directory.name}/template.json: files must be a non-empty list"
            )
        files = tuple(
            _safe_relative(value, f"{directory.name}.files[{index}]")
            for index, value in enumerate(files_raw)
        )
        if len(set(files)) != len(files):
            raise TemplateValidationError(
                f"{directory.name}/template.json: files contains duplicate paths"
            )
        config = _safe_relative(data.get("config"), f"{directory.name}.config")
        cases = _safe_relative(data.get("cases"), f"{directory.name}.cases")
        if config not in files or cases not in files:
            raise TemplateValidationError(
                f"{directory.name}/template.json: config and cases must be listed in files"
            )
        case_count = data.get("case_count")
        if isinstance(case_count, bool) or not isinstance(case_count, int) or case_count < 1:
            raise TemplateValidationError(
                f"{directory.name}/template.json: case_count must be a positive integer"
            )
        return TemplateInfo(
            name=name,
            title=_required_string(data, "title", directory.name),
            description=_required_string(data, "description", directory.name),
            config=config,
            cases=cases,
            case_count=case_count,
            files=files,
        )

    def list_templates(self) -> Sequence[TemplateInfo]:
        if not self._root.is_dir():
            raise TemplateValidationError("bundled template catalog is unavailable")
        templates = [
            self._metadata(directory)
            for directory in self._root.iterdir()
            if directory.is_dir()
            and not directory.name.startswith((".", "_"))
        ]
        return tuple(sorted(templates, key=lambda item: item.name))

    def read_files(self, template: TemplateInfo) -> dict[str, bytes]:
        directory = self._root.joinpath(template.name)
        files: dict[str, bytes] = {}
        for relative in template.files:
            resource = directory.joinpath(*PurePosixPath(relative).parts)
            if not resource.is_file():
                raise TemplateValidationError(
                    f"template {template.name!r} is missing declared file {relative!r}"
                )
            files[relative] = resource.read_bytes()
        return files


class TemplateCatalog:
    """Aggregate deterministic template metadata across configured providers."""

    def __init__(self, providers: Sequence[TemplateProvider] | None = None) -> None:
        self._providers = tuple(providers or (BundledTemplateProvider(),))

    def _entries(self) -> list[tuple[TemplateInfo, TemplateProvider]]:
        entries: list[tuple[TemplateInfo, TemplateProvider]] = []
        seen: dict[str, str] = {}
        for provider in self._providers:
            for template in provider.list_templates():
                previous = seen.get(template.name)
                if previous is not None:
                    raise TemplateValidationError(
                        f"duplicate template name {template.name!r} from "
                        f"{previous} and {template.source}"
                    )
                seen[template.name] = template.source
                entries.append((template, provider))
        return sorted(entries, key=lambda item: item[0].name)

    def list_templates(self) -> tuple[TemplateInfo, ...]:
        return tuple(template for template, _provider in self._entries())

    def resolve(self, name: str) -> tuple[TemplateInfo, TemplateProvider]:
        for template, provider in self._entries():
            if template.name == name:
                return template, provider
        available = ", ".join(item.name for item in self.list_templates()) or "(none)"
        raise TemplateNotFoundError(
            f"Unknown template {name!r}. Available templates: {available}"
        )


def list_templates(catalog: TemplateCatalog | None = None) -> tuple[TemplateInfo, ...]:
    """List catalog metadata without writing user files."""
    return (catalog or TemplateCatalog()).list_templates()


def validate_template(
    name: str,
    catalog: TemplateCatalog | None = None,
) -> TemplateInfo:
    """Validate a template through AgentEval's existing production loaders."""
    template, provider = (catalog or TemplateCatalog()).resolve(name)
    files = provider.read_files(template)
    with tempfile.TemporaryDirectory(prefix="agenteval-template-") as temporary:
        root = Path(temporary)
        for relative, content in files.items():
            target = root.joinpath(*PurePosixPath(relative).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        config_path = root.joinpath(*PurePosixPath(template.config).parts)
        cases_path = root.joinpath(*PurePosixPath(template.cases).parts)
        try:
            registry = load_agent_registry(config_path)
            cases = load_test_cases(cases_path)
        except (OSError, TypeError, ValueError, ImportError) as exc:
            raise TemplateValidationError(
                f"Template {name!r} failed AgentEval schema validation: {exc}"
            ) from exc
        if not cases:
            raise TemplateValidationError(f"Template {name!r} contains no evaluation cases")
        if len(cases) != template.case_count:
            raise TemplateValidationError(
                f"Template {name!r} declares {template.case_count} cases but contains "
                f"{len(cases)}"
            )
        configured_cases = {config.golden_suite.as_posix() for config in registry.values()}
        if template.cases not in configured_cases:
            raise TemplateValidationError(
                f"Template {name!r} config does not reference {template.cases!r}"
            )
    return template


def show_template(name: str, catalog: TemplateCatalog | None = None) -> str:
    """Render metadata and UTF-8 starter files without installing them."""
    template, provider = (catalog or TemplateCatalog()).resolve(name)
    files = provider.read_files(template)
    lines = [
        f"{template.title} ({template.name})",
        template.description,
        f"Source: {template.source}",
        f"Cases: {template.case_count}",
    ]
    for relative in template.files:
        try:
            content = files[relative].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TemplateValidationError(
                f"Template {name!r} file {relative!r} is not UTF-8 text"
            ) from exc
        lines.extend(["", f"--- {relative} ---", content.rstrip()])
    return "\n".join(lines) + "\n"


def install_template(
    name: str,
    output: str | Path,
    *,
    force: bool = False,
    catalog: TemplateCatalog | None = None,
) -> tuple[Path, ...]:
    """Install managed files after a complete conflict and schema preflight."""
    selected_catalog = catalog or TemplateCatalog()
    template, provider = selected_catalog.resolve(name)
    validate_template(name, selected_catalog)
    files = provider.read_files(template)
    root = Path(output)
    targets = {
        relative: root.joinpath(*PurePosixPath(relative).parts)
        for relative in template.files
    }
    conflicts = [target for target in targets.values() if target.exists()]
    if conflicts and not force:
        formatted = ", ".join(str(path) for path in conflicts)
        raise TemplateInstallError(
            f"Template installation would overwrite existing file(s): {formatted}. "
            "Pass --force to overwrite template-managed files."
        )
    written: list[Path] = []
    for relative in template.files:
        try:
            content = files[relative].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TemplateValidationError(
                f"Template {name!r} file {relative!r} is not UTF-8 text"
            ) from exc
        written.append(atomic_write_text(targets[relative], content))
    return tuple(written)
