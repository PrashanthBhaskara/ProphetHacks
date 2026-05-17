"""Kalshi credential helpers for read-only authenticated API calls."""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any

from .config import load_local_env


def kalshi_credential_status() -> dict[str, Any]:
    load_local_env()
    return {
        "api_key_id_present": bool(os.environ.get("KALSHI_API_KEY_ID")),
        "demo_api_key_id_present": bool(os.environ.get("KALSHI_DEMO_API_KEY_ID")),
        "private_key_b64_present": bool(os.environ.get("KALSHI_PRIVATE_KEY_B64")),
        "private_key_pem_present": bool(os.environ.get("KALSHI_PRIVATE_KEY")),
        "private_key_path_present": bool(os.environ.get("KALSHI_PRIVATE_KEY_PATH")),
        "can_sign_requests": can_sign_kalshi_requests(),
        "base_url": os.environ.get("KALSHI_BASE_URL") or "https://api.elections.kalshi.com",
    }


def can_sign_kalshi_requests() -> bool:
    return bool(_kalshi_api_key_id() and _private_key_pem_bytes())


def kalshi_auth_headers(method: str, path: str) -> dict[str, str]:
    """Return Kalshi RSA-PSS auth headers.

    `path` must be the API path only, for example
    `/trade-api/v2/markets/KX...`, not a full URL.
    """
    load_local_env()
    api_key_id = _kalshi_api_key_id()
    key_bytes = _private_key_pem_bytes()
    if not api_key_id or not key_bytes:
        return {}

    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    private_key = serialization.load_pem_private_key(
        key_bytes,
        password=None,
        backend=default_backend(),
    )
    timestamp = str(int(time.time() * 1000))
    msg = timestamp + method.upper() + path
    signature = private_key.sign(
        msg.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY": api_key_id,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("utf-8"),
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
        "Content-Type": "application/json",
    }


def _private_key_pem_bytes() -> bytes | None:
    raw = os.environ.get("KALSHI_PRIVATE_KEY_B64")
    if raw:
        try:
            return base64.b64decode(raw)
        except Exception:  # noqa: BLE001 - status helper should not throw.
            return None
    pem = os.environ.get("KALSHI_PRIVATE_KEY")
    if pem:
        return pem.replace("\\n", "\n").encode("utf-8")
    path = os.environ.get("KALSHI_PRIVATE_KEY_PATH")
    if path and Path(path).expanduser().exists():
        return Path(path).expanduser().read_bytes()
    return None


def _kalshi_api_key_id() -> str | None:
    if _env_bool("KALSHI_USE_DEMO", False):
        return os.environ.get("KALSHI_DEMO_API_KEY_ID") or os.environ.get("KALSHI_API_KEY_ID")
    return os.environ.get("KALSHI_API_KEY_ID") or os.environ.get("KALSHI_DEMO_API_KEY_ID")


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
