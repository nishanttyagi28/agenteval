"""Per-model USD/1M-token pricing table for cost estimation (§Tier 5).

Best-effort estimates for relative cost tracking across models, not
authoritative billing figures -- the same disclaimer that already applied to
the Groq-only constants this table generalizes. Update entries as providers
change list prices.
"""

from __future__ import annotations

# model name -> (input_usd_per_1m_tokens, output_usd_per_1m_tokens)
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "claude-opus-4-8": (15.00, 75.00),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
}


def get_pricing(model: str | None, *, default: tuple[float, float]) -> tuple[float, float]:
    """Look up (input, output) USD/1M pricing for ``model``.

    Falls back to ``default`` when ``model`` is ``None`` or not in the table
    -- this preserves ``compute_cost_usd``'s original Groq-only behavior for
    every existing caller that doesn't pass a model.
    """
    if model is None:
        return default
    return MODEL_PRICING.get(model, default)
