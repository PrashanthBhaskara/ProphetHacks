"""Small Prophet Arena API client."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests

from .config import load_local_env


DEFAULT_BASE_URL = "https://api.aiprophet.dev"


def prophet_api_status() -> dict[str, Any]:
    load_local_env()
    return {
        "api_key_present": bool(os.environ.get("PA_SERVER_API_KEY")),
        "base_url": _base_url(),
    }


def health() -> dict[str, Any]:
    response = requests.get(f"{_base_url()}/health", timeout=10)
    response.raise_for_status()
    try:
        return response.json()
    except ValueError:
        return {"status": response.text.strip() or "ok"}


def forecast_events(*, status: str = "open") -> list[dict[str, Any]]:
    response = requests.get(
        f"{_base_url()}/forecast/events",
        params={"status": status},
        headers=_headers(),
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    return data if isinstance(data, list) else data.get("events", [])


def forecast_scores() -> list[dict[str, Any]]:
    response = requests.get(
        f"{_base_url()}/forecast/scores",
        headers=_headers(),
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    return data if isinstance(data, list) else data.get("scores", [])


def write_events(path: Path, *, status: str = "open") -> list[dict[str, Any]]:
    events = forecast_events(status=status)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(events, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return events


def _headers() -> dict[str, str]:
    load_local_env()
    key = os.environ.get("PA_SERVER_API_KEY")
    if not key:
        raise RuntimeError("PA_SERVER_API_KEY is not configured")
    return {"X-API-Key": key}


def _base_url() -> str:
    load_local_env()
    return (os.environ.get("PA_SERVER_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
