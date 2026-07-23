from __future__ import annotations

from pathlib import Path

import pytest

from agenteval.core.registry import load_agent_registry
from agenteval.core.schema import load_test_cases
from agenteval.core.template_catalog import (
    BundledTemplateProvider,
    TemplateInstallError,
    install_template,
    list_templates,
    show_template,
    validate_template,
)


EXPECTED = ["coding-agent", "customer-support", "rag-assistant"]


def test_bundled_catalog_is_deterministic_and_realistic():
    templates = list_templates()
    assert [item.name for item in templates] == EXPECTED
    assert all(item.source == "bundled" for item in templates)
    assert all(item.case_count == 7 for item in templates)
    assert all(item.description for item in templates)


def test_bundled_catalog_ignores_python_cache_directory(tmp_path):
    provider = BundledTemplateProvider()
    provider._root = tmp_path
    (tmp_path / "__pycache__").mkdir()

    assert provider.list_templates() == ()


@pytest.mark.parametrize("name", EXPECTED)
def test_each_template_validates_through_existing_loaders(name):
    template = validate_template(name)
    assert template.name == name


def test_show_template_includes_metadata_and_starter_files():
    rendered = show_template("rag-assistant")
    assert "RAG Assistant (rag-assistant)" in rendered
    assert "--- agents.yaml ---" in rendered
    assert "--- cases.yaml ---" in rendered
    assert "retrieved_prompt_injection" in rendered


@pytest.mark.parametrize("name", EXPECTED)
def test_installed_template_loads_from_destination(name, tmp_path):
    destination = tmp_path / name
    written = install_template(name, destination)
    assert {path.name for path in written} == {"README.md", "agents.yaml", "cases.yaml"}
    registry = load_agent_registry(destination / "agents.yaml")
    cases = load_test_cases(destination / "cases.yaml")
    assert len(registry) == 1
    assert len(cases) == 7
    assert next(iter(registry.values())).enabled is False


def test_install_preflights_every_conflict_before_writing(tmp_path):
    destination = tmp_path / "catalog"
    destination.mkdir()
    (destination / "cases.yaml").write_text("user content\n", encoding="utf-8")

    with pytest.raises(TemplateInstallError, match="would overwrite"):
        install_template("rag-assistant", destination)

    assert (destination / "cases.yaml").read_text(encoding="utf-8") == "user content\n"
    assert not (destination / "README.md").exists()
    assert not (destination / "agents.yaml").exists()


def test_force_overwrites_only_managed_files(tmp_path):
    destination = tmp_path / "catalog"
    install_template("coding-agent", destination)
    unrelated = destination / "notes.txt"
    unrelated.write_text("keep me\n", encoding="utf-8")
    (destination / "cases.yaml").write_text("stale\n", encoding="utf-8")

    install_template("coding-agent", destination, force=True)

    assert "diagnose_none_crash" in (destination / "cases.yaml").read_text(
        encoding="utf-8"
    )
    assert unrelated.read_text(encoding="utf-8") == "keep me\n"


def test_unknown_template_is_actionable():
    with pytest.raises(ValueError, match="Available templates"):
        validate_template("missing-template")
