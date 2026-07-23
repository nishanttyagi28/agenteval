from pathlib import Path
import tomllib

from agenteval import __version__
from agenteval.cli import build_parser


ROOT = Path(__file__).resolve().parents[1]


def load_pyproject() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_packaging_uses_init_version_as_single_source_of_truth():
    config = load_pyproject()

    assert "version" not in config["project"]
    assert "version" in config["project"]["dynamic"]
    assert config["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "agenteval.__version__"
    }
    assert __version__ == "0.1.0"


def test_console_script_targets_existing_cli_main():
    config = load_pyproject()

    assert config["project"]["scripts"]["agenteval"] == "agenteval.cli:main"
    assert build_parser().prog == "agenteval"


def test_default_runtime_configuration_is_included_as_package_data():
    package_data = load_pyproject()["tool"]["setuptools"]["package-data"][
        "agenteval"
    ]

    assert "agents.yaml" in package_data
    assert "baselines/*.json" in package_data
    assert "tests/golden/*.yaml" in package_data


def test_bundled_templates_are_declared_as_package_data():
    config = load_pyproject()

    assert "agenteval.templates" in config["tool"]["setuptools"]["packages"]
    assert "agenteval.templates.catalog" in config["tool"]["setuptools"]["packages"]
    assert config["tool"]["setuptools"]["package-data"][
        "agenteval.templates.catalog"
    ] == [
        "*/*.json",
        "*/*.md",
        "*/*.yaml",
    ]


def test_crewai_is_an_optional_framework_dependency():
    extras = load_pyproject()["project"]["optional-dependencies"]

    assert extras["crewai"] == ["crewai>=1,<2"]


def test_autogen_is_an_optional_framework_dependency():
    extras = load_pyproject()["project"]["optional-dependencies"]

    assert extras["autogen"] == ["autogen-agentchat>=0.7,<1"]


def test_openai_agents_is_an_optional_framework_dependency():
    extras = load_pyproject()["project"]["optional-dependencies"]

    assert extras["openai-agents"] == ["openai-agents>=0.18,<1"]
