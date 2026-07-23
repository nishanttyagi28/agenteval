"""Deterministic transcript rendering for multi-turn conversation cases (§Tier 9).

Self-contained and adapter-agnostic, mirroring ``core.trajectory``/
``core.rag_metrics``: this module only formats plain text from already
-observed (prompt, answer) pairs. It never invokes an agent and never
changes what ``adapter.run()`` expects -- ``core.runner.run_conversation_case``
calls ``adapter.run(prompt: str)`` exactly as every existing adapter already
implements it, passing the *rendered* string as ``prompt``. There is no new
adapter method or keyword argument.
"""

from __future__ import annotations

from typing import Sequence

Turn_Pair = tuple[str, str]  # (user prompt, agent answer)


def render_turn_prompt(history: Sequence[Turn_Pair], next_prompt: str) -> str:
    """Build the text sent to ``adapter.run()`` for one conversation turn.

    ``history`` holds every prior (prompt, answer) pair, oldest first. An
    empty history (the first turn) returns ``next_prompt`` completely
    unchanged, so turn 0 of a multi-turn case is byte-identical to a
    single-turn case's prompt -- no adapter can distinguish the first turn
    of a conversation from an ordinary one-shot call.

    Example (2 prior turns)::

        >>> render_turn_prompt(
        ...     [("What is the capital of France?", "Paris"),
        ...      ("What is its population?", "About 2.1 million.")],
        ...     "And its most famous landmark?",
        ... )
        'Conversation so far:
        User (turn 1): What is the capital of France?
        Assistant (turn 1): Paris
        User (turn 2): What is its population?
        Assistant (turn 2): About 2.1 million.

        User (turn 3): And its most famous landmark?'
    """
    if not history:
        return next_prompt
    lines = ["Conversation so far:"]
    for index, (user_prompt, answer) in enumerate(history, start=1):
        lines.append(f"User (turn {index}): {user_prompt}")
        lines.append(f"Assistant (turn {index}): {answer}")
    lines.append("")
    lines.append(f"User (turn {len(history) + 1}): {next_prompt}")
    return "\n".join(lines)


def render_full_transcript(turns: Sequence[Turn_Pair]) -> tuple[str, str]:
    """Return ``(joined_prompts, joined_answers)`` for whole-conversation scoring.

    Used for the case-level goal-completion verdict: every turn's prompt and
    every turn's answer, each joined into one multi-line string, so the
    existing ``check_correctness``/``check_hallucination`` functions (which
    take a single ``prompt``/``final_answer`` pair) can judge the *entire*
    conversation's outcome rather than only its last message, without any
    change to their own signatures.
    """
    prompt_lines = [f"User (turn {i}): {p}" for i, (p, _a) in enumerate(turns, start=1)]
    answer_lines = [f"Assistant (turn {i}): {a}" for i, (_p, a) in enumerate(turns, start=1)]
    return "\n".join(prompt_lines), "\n".join(answer_lines)


__all__ = ["render_turn_prompt", "render_full_transcript"]
