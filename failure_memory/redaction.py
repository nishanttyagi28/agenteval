"""Deterministic recursive redaction before persistence.

This is risk reduction, not a complete DLP system. Prefer content capture
off by default and treat redacted storage as still sensitive.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REDACTION_VERSION = 2

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
        # Common leaky diagnostic payload keys in tool results
        "debug",
        "credential",
        "credentials",
    }
)

# Value patterns — keep simple and linear to avoid catastrophic backtracking.
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9\-._~+/]+=*")
_BASIC_AUTH_RE = re.compile(r"(?i)\bBasic\s+[A-Za-z0-9+/=]{8,}")
_OPENAI_RE = re.compile(r"\bsk-[A-Za-z0-9]{16,}\b")
_GITHUB_RE = re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")
_SLACK_RE = re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")
_AWS_KEY_RE = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
_ANTHROPIC_RE = re.compile(r"\bsk-ant-[A-Za-z0-9\-_]{16,}\b")
# Value after key= may include hyphens; stop at whitespace only.
_GENERIC_SECRET_RE = re.compile(
    r"(?i)\b(?:api[_-]?key|secret|token|password|authorization|passwd)\s*[:=]\s*['\"]?(\S{8,})"
)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\w)"
)
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    r"[\s\S]*?"
    r"-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
)

# Process-local configurable custom patterns (tests / advanced users).
_CUSTOM_PATTERNS: list[re.Pattern[str]] = []


def set_custom_secret_patterns(patterns: Sequence[str]) -> None:
    """Register additional secret regexes (replaces prior custom set)."""
    global _CUSTOM_PATTERNS
    compiled: list[re.Pattern[str]] = []
    for p in patterns:
        compiled.append(re.compile(p))
    _CUSTOM_PATTERNS = compiled


def clear_custom_secret_patterns() -> None:
    global _CUSTOM_PATTERNS
    _CUSTOM_PATTERNS = []


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
    return compact.endswith(("token", "secret", "password", "apikey", "privatekey"))


def redact_string(text: str) -> tuple[str, bool]:
    """Return (redacted_text, changed)."""
    if not text:
        return text, False
    original = text
    text = _PRIVATE_KEY_RE.sub(PLACEHOLDER_PRIVATE_KEY, text)
    text = _BEARER_RE.sub(f"Bearer {PLACEHOLDER_SECRET}", text)
    text = _BASIC_AUTH_RE.sub(f"Basic {PLACEHOLDER_SECRET}", text)
    text = _OPENAI_RE.sub(PLACEHOLDER_SECRET, text)
    text = _ANTHROPIC_RE.sub(PLACEHOLDER_SECRET, text)
    text = _GITHUB_RE.sub(PLACEHOLDER_SECRET, text)
    text = _SLACK_RE.sub(PLACEHOLDER_SECRET, text)
    text = _AWS_KEY_RE.sub(PLACEHOLDER_SECRET, text)
    text = _GENERIC_SECRET_RE.sub(
        lambda m: m.group(0)[: m.start(1) - m.start(0)] + PLACEHOLDER_SECRET, text
    )
    for pat in _CUSTOM_PATTERNS:
        text = pat.sub(PLACEHOLDER_SECRET, text)
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


def find_raw_secrets_in_bytes(data: bytes, secrets: Iterable[str]) -> list[str]:
    """Return which secret strings appear as raw UTF-8 substrings in ``data``."""
    hits: list[str] = []
    for secret in secrets:
        if not secret:
            continue
        if secret.encode("utf-8") in data:
            hits.append(secret)
    return hits


def find_raw_secrets_in_tree(root: str | Path, secrets: Iterable[str]) -> list[str]:
    """Scan all files under ``root`` for raw secret substrings.

    Returns human-readable ``path: secret`` hit strings (empty = clean).
    """
    root_path = Path(root)
    secret_list = [s for s in secrets if s]
    hits: list[str] = []
    if not root_path.exists():
        return hits
    paths: list[Path]
    if root_path.is_file():
        paths = [root_path]
    else:
        paths = [p for p in root_path.rglob("*") if p.is_file()]
    for path in paths:
        try:
            data = path.read_bytes()
        except OSError:
            continue
        for secret in find_raw_secrets_in_bytes(data, secret_list):
            hits.append(f"{path}: {secret}")
    return hits
