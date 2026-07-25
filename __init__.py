"""AgentEval — CI-integrated evaluation harness for LLM agents."""

from agenteval.compat import AgentEvalDeprecationWarning, warn_deprecated

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "AgentEvalDeprecationWarning",
    "warn_deprecated",
]
