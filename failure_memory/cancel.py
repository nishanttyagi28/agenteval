"""Cooperative cancellation for long replay and minimization loops."""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass
class CancellationToken:
    """Thread-safe cooperative cancellation flag.

    Callers check ``is_cancelled`` between attempts. ``cancel()`` is idempotent.
    """

    _event: threading.Event = None  # type: ignore[assignment]
    reason: str = "cancelled"

    def __post_init__(self) -> None:
        if self._event is None:
            object.__setattr__(self, "_event", threading.Event())

    def cancel(self, reason: str = "cancelled") -> None:
        self.reason = reason
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise CancellationError(self.reason)


class CancellationError(Exception):
    """Raised when a cooperative cancellation token is set."""
