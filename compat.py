"""Compatibility and deprecation helpers for AgentEval's public surfaces."""

from __future__ import annotations

import warnings


class AgentEvalDeprecationWarning(FutureWarning):
    """Visible warning category for behavior scheduled to change."""


def warn_deprecated(
    feature: str,
    *,
    removal: str,
    alternative: str | None = None,
    stacklevel: int = 2,
) -> None:
    """Emit one actionable warning using AgentEval's public warning category."""
    message = f"{feature} is deprecated and is scheduled for removal in {removal}."
    if alternative:
        message += f" Use {alternative} instead."
    warnings.warn(message, AgentEvalDeprecationWarning, stacklevel=stacklevel)


__all__ = ["AgentEvalDeprecationWarning", "warn_deprecated"]
