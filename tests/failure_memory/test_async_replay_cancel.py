"""Async replay adapters and cooperative cancellation."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agenteval.failure_memory.cancel import CancellationToken
from agenteval.failure_memory.minimize import minimize_payload
from agenteval.failure_memory.replay import (
    AsyncFakeReplayAdapter,
    FakeReplayAdapter,
    ReplayCase,
    ReplayOutcome,
    run_replay,
    run_replay_async,
)
from agenteval.failure_memory.store import SQLiteFailureMemoryStore


def _case() -> ReplayCase:
    return ReplayCase(
        candidate_id="cand_async",
        fingerprint="e" * 64,
        agent_name="refund-agent",
        prompt="refund",
        attributes={
            "must_call_tools": ["lookup_order", "issue_refund"],
            "tools_called": ["cancel_order"],
        },
        expected_category="wrong_tool",
    )


def test_async_adapter_reproduced(tmp_path: Path):
    store = SQLiteFailureMemoryStore(tmp_path / "fm.db")
    rep = run_replay(
        store,
        adapter=AsyncFakeReplayAdapter(mode="reproduce"),
        attempts=4,
        threshold=0.75,
        case_override=_case(),
        idempotency_key="async-ok",
    )
    assert rep.outcome == ReplayOutcome.reproduced
    rep2 = run_replay(
        store,
        adapter=AsyncFakeReplayAdapter(mode="reproduce"),
        attempts=4,
        case_override=_case(),
        idempotency_key="async-ok",
    )
    assert rep2.replay_id == rep.replay_id
    store.close()


def test_async_adapter_timeout(tmp_path: Path):
    store = SQLiteFailureMemoryStore(tmp_path / "fm.db")
    rep = run_replay(
        store,
        adapter=AsyncFakeReplayAdapter(mode="timeout"),
        attempts=2,
        timeout_s=0.2,
        case_override=_case(),
    )
    assert rep.outcome == ReplayOutcome.infrastructure_error
    assert rep.diagnostics.get("infra_errors", 0) >= 1
    store.close()


def test_async_adapter_infra(tmp_path: Path):
    store = SQLiteFailureMemoryStore(tmp_path / "fm.db")
    rep = run_replay(
        store,
        adapter=AsyncFakeReplayAdapter(mode="infra"),
        attempts=3,
        case_override=_case(),
    )
    assert rep.outcome == ReplayOutcome.infrastructure_error
    store.close()


def test_async_entrypoint(tmp_path: Path):
    store = SQLiteFailureMemoryStore(tmp_path / "fm.db")

    async def _go():
        return await run_replay_async(
            store,
            adapter=AsyncFakeReplayAdapter(mode="reproduce"),
            attempts=3,
            threshold=0.6,
            case_override=_case(),
        )

    rep = asyncio.run(_go())
    assert rep.outcome == ReplayOutcome.reproduced
    store.close()


def test_replay_cancellation(tmp_path: Path):
    store = SQLiteFailureMemoryStore(tmp_path / "fm.db")
    token = CancellationToken()
    # Cancel immediately so loop stops early
    token.cancel("test-stop")
    rep = run_replay(
        store,
        adapter=FakeReplayAdapter(mode="reproduce"),
        attempts=10,
        case_override=_case(),
        cancellation_token=token,
    )
    assert rep.outcome == ReplayOutcome.cancelled
    row = store.get_replay_run(rep.replay_id)
    assert row is not None
    assert row["outcome"] == "cancelled"
    assert row["ended_at"]  # not left running
    store.close()


def test_minimize_cancellation_returns_best(tmp_path: Path):
    store = SQLiteFailureMemoryStore(tmp_path / "fm.db")
    token = CancellationToken()
    token.cancel("stop-min")
    payload = {
        "prompt": "refund",
        "messages": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
        "tool_trace": [{"name": "cancel_order"}],
        "must_call_tools": ["lookup_order", "issue_refund"],
        "debug": "x",
    }
    result = minimize_payload(
        store,
        source_candidate_id="cand_x",
        payload=payload,
        expected_category="wrong_tool",
        expected_fingerprint=None,
        agent_name="a",
        adapter=FakeReplayAdapter(mode="reproduce"),
        max_attempts=20,
        replay_attempts=1,
        threshold=0.5,
        cancellation_token=token,
    )
    assert result.cancelled is True
    row = store.get_minimized_case(result.minimization_id)
    assert row is not None
    assert row["approval_state"] == "cancelled"
    # best valid result still stored (payload present)
    assert row["minimized_payload_json"]
    store.close()
