"""Nonsecret credential metadata helpers."""

from __future__ import annotations

import hashlib
from typing import Any


def key_fingerprint(value: str | None) -> str | None:
    """Return a stable nonsecret fingerprint for audit logs."""
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def api_key_metadata(
    *,
    value: str | None,
    env_name: str | None,
    expected_prefix: str | None = None,
) -> dict[str, Any]:
    """Summarize an API key without exposing secret bytes."""
    return {
        "api_key_env": env_name,
        "key_present": bool(value),
        "key_length": len(value) if value else 0,
        "prefix_valid": None if not expected_prefix else bool(value and value.startswith(expected_prefix)),
        "api_key_fingerprint": key_fingerprint(value),
    }
