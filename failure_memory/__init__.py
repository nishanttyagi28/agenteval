"""Agent Failure Memory — production failures → human-approved regression tests.

Local-first, deterministic, privacy-preserving. Not a hosted observability platform.
"""

from __future__ import annotations

from agenteval.failure_memory.recorder import FailureMemoryRecorder, ingest_jsonl
from agenteval.failure_memory.schema import TraceEnvelope, TraceStatus
from agenteval.failure_memory.service import FailureMemoryService
from agenteval.failure_memory.store import DEFAULT_DB_PATH, ENV_DB_PATH, open_store

__all__ = [
    "DEFAULT_DB_PATH",
    "ENV_DB_PATH",
    "FailureMemoryRecorder",
    "FailureMemoryService",
    "TraceEnvelope",
    "TraceStatus",
    "ingest_jsonl",
    "open_store",
]
