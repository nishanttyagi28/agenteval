"""Deterministic recursive redaction before persistence.

This is risk reduction, not a complete DLP system. Prefer opt-out content
capture and treat redacted storage as still sensitive.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Mapping, MutableMapping, Sequence

REDACTION_VERSION = 1

PLACEHOLDER_SECRET = "[REDACTED_SECRET]"
PLACEHOLDER_EMAIL = "[REDACTED_EMAIL]"
PLACEHOLDER_PHONE = "[REDACTED_PHONE]"
PLACEHOLDER_PRIVATE_KEY = "[REDACTED_PRIVATE_KEY]"

# Sensitive key names (case-insensitive, with common separators normalized).
_SENSITIVE_KEYS = frozenset(
    {
        "authorization",
        "auth",
        "api_key",
        "apikey",
        "api-key",
        "password",
        "passwd",
        "secret",
        "access_token",
        "accesstoken",
        "refresh_token",
        "refreshtoken",
        "id_token",
        "cookie",
        "set_cookie",
        "set-cookie",
        "private_key",
        "privatekey",
        "client_secret",
        "clientsecret",
        "x_api_key",
        "x-api-key",
        "token",
        "bearer",
        "aws_secret_access_key",
        "aws_access_key_id",
    }
)

# Value patterns — keep simple and linear to avoid catastrophic backtracking.
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9\-._~+/]+=*")
_OPENAI_RE = re.compile(r"\bsk-[A-Za-z0-9]{16,}\b")
_GITHUB_RE = re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")
_SLACK_RE = re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")
_AWS_KEY_RE = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
_GENERIC_SECRET_RE = re.compile(
    r"(?i)\b(?:api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?([^\s'\"]{8,})"
)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# Simple E.164-ish / US-style phones; not exhaustive.
_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\w)"
)
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
)


def _normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def _key_is_sensitive(key: str) -> bool:
    raw = key.lower().replace("-", "_")
    if raw in _SENSITIVE_KEYS:
        return True
    compact = _normalize_key(key)
    sensitive_compact = {_normalize_key(k) for k in _SENSITIVE_KEYS}
    if compact in sensitive_compact:
        return True
    # Heuristic: *token / *secret / *password suffixes
    return compact.endswith(("token", "secret", "password", "apikey", "privatekey"))


def redact_string(text: str) -> tuple[str, bool]:
    """Return (redacted_text, changed)."""
    if not text:
        return text, False
    original = text
    text = _PRIVATE_KEY_RE.sub(PLACEHOLDER_PRIVATE_KEY, text)
    text = _BEARER_RE.sub(f"Bearer {PLACEHOLDER_SECRET}", text)
    text = _OPENAI_RE.sub(PLACEHOLDER_SECRET, text)
    text = _GITHUB_RE.sub(PLACEHOLDER_SECRET, text)
    text = _SLACK_RE.sub(PLACEHOLDER_SECRET, text)
    text = _AWS_KEY_RE.sub(PLACEHOLDER_SECRET, text)
    text = _GENERIC_SECRET_RE.sub(
        lambda m: m.group(0).split(m.group(1))[0] + PLACEHOLDER_SECRET, text
    )
    text = _EMAIL_RE.sub(PLACEHOLDER_EMAIL, text)
    text = _PHONE_RE.sub(PLACEHOLDER_PHONE, text)
    return text, text != original


def redact_value(value: Any, *, key: str | None = None) -> tuple[Any, bool]:
    """Recursively redact. Never mutates caller-owned objects."""
    changed = False
    if key is not None and _key_is_sensitive(key):
        if isinstance(value, str) and "PRIVATE KEY" in value:
            return PLACEHOLDER_PRIVATE_KEY, True
        return PLACEHOLDER_SECRET, True

    if isinstance(value, str):
        return redact_string(value)

    if isinstance(value, Mapping):
        out: dict[Any, Any] = {}
        for k, v in value.items():
            k_str = k if isinstance(k, str) else str(k)
            nv, ch = redact_value(v, key=k_str)
            out[k] = nv
            changed = changed or ch
        return out, changed

    if isinstance(value, list):
        out_list = []
        for item in value:
            nv, ch = redact_value(item, key=key)
            out_list.append(nv)
            changed = changed or ch
        return out_list, changed

    if isinstance(value, tuple):
        items = []
        for item in value:
            nv, ch = redact_value(item, key=key)
            items.append(nv)
            changed = changed or ch
        return tuple(items), changed

    return value, False


def redact_mapping(data: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    """Deep-copy and redact a mapping."""
    redacted, changed = redact_value(copy.deepcopy(dict(data)))
    assert isinstance(redacted, dict)
    return redacted, changed
