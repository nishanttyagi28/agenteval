"""Shared stdlib-only text helpers for the Indic evaluator checks."""

from __future__ import annotations

import re
from typing import Iterable

REPLACEMENT_CHAR = "�"
_ESCAPED_UNICODE_RE = re.compile(r"\\u[0-9a-fA-F]{4}")

_DEVANAGARI_START = 0x0900
_DEVANAGARI_END = 0x097F

# Common untranslated tech/proper-noun terms that legitimately appear inside
# an otherwise Devanagari (or otherwise non-Latin) answer without indicating
# script drift. Overridable per case via ground_truth["allow_terms"].
DEFAULT_ALLOW_TERMS: tuple[str, ...] = (
    "AI", "API", "GPU", "CPU", "OTP", "PIN", "SMS", "URL", "ID",
    "CEO", "CFO", "PDF", "OK", "UPI", "EMI", "Wi-Fi", "IP", "SIM",
)


def is_devanagari(ch: str) -> bool:
    return _DEVANAGARI_START <= ord(ch) <= _DEVANAGARI_END


def is_latin(ch: str) -> bool:
    return ch.isascii() and ch.isalpha()


def strip_allow_terms(text: str, allow_terms: Iterable[str]) -> str:
    """Blank out literal allow-listed terms (word-boundary, case-insensitive)."""
    for term in sorted((t for t in allow_terms if t), key=len, reverse=True):
        text = re.sub(re.escape(term), " ", text, flags=re.IGNORECASE)
    return text


def has_mojibake(text: str) -> bool:
    """True when text carries a Unicode replacement char or an escaped-\\u literal."""
    return REPLACEMENT_CHAR in text or bool(_ESCAPED_UNICODE_RE.search(text))
