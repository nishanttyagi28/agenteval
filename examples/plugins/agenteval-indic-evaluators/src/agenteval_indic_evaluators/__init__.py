"""Deterministic Indic-language evaluator plugins for AgentEval.

Registers three ``agenteval.evaluators`` entry points: ``script_consistency``,
``transliteration_stability``, and ``tool_arg_encoding``. See README.md for
each checker's ``ground_truth`` shape.
"""

from __future__ import annotations

from .script_consistency import evaluate as evaluate_script_consistency
from .tool_arg_encoding import evaluate as evaluate_tool_arg_encoding
from .transliteration_stability import evaluate as evaluate_transliteration_stability

__all__ = [
    "evaluate_script_consistency",
    "evaluate_transliteration_stability",
    "evaluate_tool_arg_encoding",
]
