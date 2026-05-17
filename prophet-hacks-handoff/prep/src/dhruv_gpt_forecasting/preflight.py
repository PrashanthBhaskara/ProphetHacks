"""No-credit readiness checks for live Prophet Arena forecasts."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import requests

from .arena_agent import forecast_arena_event
from .config import ForecastConfig, load_config, load_local_env, resolve_api_key
from .kalshi_auth import kalshi_credential_status
from .key_utils import api_key_metadata
from .openrouter import GEMINI_MODELS_URL, OPENROUTER_AUTH_URL, OPENROUTER_MODELS_URL
from .prophet_api import health as prophet_health
from .prophet_api import prophet_api_status
from .vendor_evidence import vendor_env_status


def run_preflight(
    *,
    config: ForecastConfig | None = None,
    offline: bool = False,
    timeout_seconds: float = 10.0,
    spend_gpt: bool = False,
) -> dict[str, Any]:
    """Run sanitized checks. Default mode never calls GPT completions."""
    started = time.monotonic()
    load_local_env()
    cfg = config or load_config()
    key, key_env = resolve_api_key(cfg.model)
    llm = _llm_status(
        cfg,
        key=key,
        key_env=key_env,
        offline=offline,
        timeout_seconds=timeout_seconds,
    )
    prophet = _prophet_status(offline=offline, timeout_seconds=timeout_seconds)
    kalshi = kalshi_credential_status()
    cache = _cache_status(cfg)
    offline_prediction = _offline_prediction_status(cfg)
    live_sources = _live_source_status()
    gpt_smoke = None
    if spend_gpt:
        gpt_smoke = _gpt_smoke_status(cfg, timeout_seconds=timeout_seconds)

    checks = {
        "llm_key_present": bool(llm["key"].get("key_present")),
        "llm_prefix_valid": bool(llm["key"].get("prefix_valid")),
        "llm_auth_ok": True if offline else llm.get("auth", {}).get("ok") is True,
        "selected_model_available": True if offline else llm.get("model_availability", {}).get("available") is True,
        "cache_writable": cache.get("writable") is True,
        "offline_prediction_valid": offline_prediction.get("valid") is True,
    }
    if gpt_smoke is not None:
        checks["gpt_smoke_ok"] = gpt_smoke.get("ok") is True
    result = {
        "ok": all(checks.values()),
        "mode": "offline" if offline else "network",
        "spend_gpt": spend_gpt,
        "elapsed_seconds": time.monotonic() - started,
        "checks": checks,
        "llm": llm,
        "prophet_arena": prophet,
        "kalshi": kalshi,
        "live_evidence_sources": live_sources,
        "cache": cache,
        "offline_prediction": offline_prediction,
    }
    if gpt_smoke is not None:
        result["gpt_smoke"] = gpt_smoke
    return result


def _llm_status(
    cfg: ForecastConfig,
    *,
    key: str | None,
    key_env: str | None,
    offline: bool,
    timeout_seconds: float,
) -> dict[str, Any]:
    if cfg.model.provider == "gemini":
        return _gemini_status(
            cfg,
            key=key,
            key_env=key_env,
            offline=offline,
            timeout_seconds=timeout_seconds,
        )
    return _openrouter_status(
        cfg,
        key=key,
        key_env=key_env,
        offline=offline,
        timeout_seconds=timeout_seconds,
    )


def _openrouter_status(
    cfg: ForecastConfig,
    *,
    key: str | None,
    key_env: str | None,
    offline: bool,
    timeout_seconds: float,
) -> dict[str, Any]:
    status: dict[str, Any] = {
        "provider": "openrouter",
        "selected_model": cfg.model.model,
        "native_search_grounding": {
            "enabled": cfg.model.native_search_grounding_enabled,
            "live_only": cfg.model.native_search_grounding_live_only,
            "engine": cfg.model.search_grounding_engine,
            "max_results": cfg.model.search_grounding_max_results,
            "max_total_results": cfg.model.search_grounding_max_total_results,
        },
        "key": api_key_metadata(value=key, env_name=key_env, expected_prefix="sk-or-"),
        "auth": {"checked": False},
        "model_availability": {"checked": False, "available": None},
    }
    if offline or not key:
        return status
    status["auth"] = _safe_get(
        OPENROUTER_AUTH_URL,
        headers={"Authorization": f"Bearer {key}"},
        timeout_seconds=timeout_seconds,
        include_json_keys=("usage", "limit", "limit_remaining", "is_free_tier", "rate_limit", "label"),
    )
    models = _safe_get(
        OPENROUTER_MODELS_URL,
        headers={},
        timeout_seconds=timeout_seconds,
        include_json_keys=(),
    )
    available = None
    if models.get("ok") and isinstance(models.get("json"), dict):
        rows = models["json"].get("data") or []
        if isinstance(rows, list):
            ids = {str(row.get("id")) for row in rows if isinstance(row, dict)}
            available = cfg.model.model in ids
    status["model_availability"] = {
        "checked": True,
        "ok": models.get("ok"),
        "status_code": models.get("status_code"),
        "available": available,
        "error": models.get("error"),
    }
    return status


def _gemini_status(
    cfg: ForecastConfig,
    *,
    key: str | None,
    key_env: str | None,
    offline: bool,
    timeout_seconds: float,
) -> dict[str, Any]:
    status: dict[str, Any] = {
        "provider": "gemini",
        "selected_model": cfg.model.model,
        "native_search_grounding": {
            "enabled": cfg.model.native_search_grounding_enabled,
            "live_only": cfg.model.native_search_grounding_live_only,
            "engine": "google_search",
            "max_results": cfg.model.search_grounding_max_results,
            "max_total_results": cfg.model.search_grounding_max_total_results,
        },
        "key": api_key_metadata(value=key, env_name=key_env, expected_prefix="AIza"),
        "auth": {"checked": False},
        "model_availability": {"checked": False, "available": None},
    }
    if offline or not key:
        return status
    models = _safe_get(
        GEMINI_MODELS_URL,
        headers={"x-goog-api-key": key},
        timeout_seconds=timeout_seconds,
        include_json_keys=(),
    )
    available = None
    if models.get("ok") and isinstance(models.get("json"), dict):
        rows = models["json"].get("models") or models["json"].get("data") or []
        if isinstance(rows, list):
            ids = {
                str(row.get("name", "")).removeprefix("models/")
                for row in rows
                if isinstance(row, dict)
            }
            available = cfg.model.model in ids
    status["auth"] = {
        "checked": True,
        "ok": models.get("ok"),
        "status_code": models.get("status_code"),
        "error": models.get("error"),
    }
    status["model_availability"] = {
        "checked": True,
        "ok": models.get("ok"),
        "status_code": models.get("status_code"),
        "available": available,
        "error": models.get("error"),
    }
    return status


def _prophet_status(*, offline: bool, timeout_seconds: float) -> dict[str, Any]:
    status = prophet_api_status()
    status["health"] = {"checked": False}
    if offline:
        return status
    try:
        status["health"] = {"checked": True, "ok": True, "response": prophet_health()}
    except Exception as exc:  # noqa: BLE001 - preflight should report all failures.
        status["health"] = {"checked": True, "ok": False, "error": _safe_error(exc)}
    status["health"]["timeout_seconds"] = timeout_seconds
    return status


def _live_source_status() -> dict[str, Any]:
    source_groups = {
        "fred": [["FRED_API_KEY"]],
        "bea": [["BEA_API_KEY"]],
        "eia": [["EIA_API_KEY"]],
        "polygon": [["POLYGON_API_KEY"]],
        "oddspipe": [["ODDSPIPE_API_KEY"], ["ODDSPIPE_API_URL"]],
        "reddit": [["REDDIT_USER_AGENT"]],
    }
    status = {}
    for source, groups in source_groups.items():
        all_envs = [name for group in groups for name in group]
        configured = all(any(os.environ.get(name) for name in group) for group in groups)
        if source == "reddit":
            configured = True
        status[source] = {
            "configured": configured,
            "required_env_groups": groups,
            "present_envs": [name for name in all_envs if os.environ.get(name)],
        }
        if source == "reddit" and not status[source]["present_envs"]:
            status[source]["default_user_agent"] = "ProphetHacksGPTForecasting/0.1"
    status["wrds"] = vendor_env_status("wrds")
    status["lseg"] = vendor_env_status("lseg")
    return status


def _cache_status(cfg: ForecastConfig) -> dict[str, Any]:
    log_dir = Path(cfg.budget.log_dir)
    probe = log_dir / ".preflight_write_test"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return {"path": str(log_dir), "writable": True}
    except Exception as exc:  # noqa: BLE001
        return {"path": str(log_dir), "writable": False, "error": _safe_error(exc)}


def _offline_prediction_status(cfg: ForecastConfig) -> dict[str, Any]:
    event = {
        "event_ticker": "preflight-demo",
        "market_ticker": "preflight-demo",
        "title": "Will the preflight forecast return valid probabilities?",
        "description": "Synthetic offline validation event.",
        "category": "Politics",
        "rules": "Resolves Yes if this synthetic event is marked true.",
        "close_time": "2026-12-31T23:59:59Z",
        "outcomes": ["YES", "NO"],
    }
    try:
        forecast = forecast_arena_event(
            event,
            config=cfg,
            use_gpt=False,
            use_live_data=False,
            deadline_seconds=30,
        )
        probs = forecast.probabilities
        valid = set(probs) == {"YES", "NO"} and abs(sum(probs.values()) - 1.0) < 1e-9
        return {
            "valid": valid,
            "source": forecast.source,
            "probability_sum": sum(probs.values()),
            "outcomes": list(probs),
        }
    except Exception as exc:  # noqa: BLE001
        return {"valid": False, "error": _safe_error(exc)}


def _gpt_smoke_status(cfg: ForecastConfig, *, timeout_seconds: float) -> dict[str, Any]:
    old_timeout = os.environ.get("OPENROUTER_TIMEOUT_SECONDS")
    old_gemini_timeout = os.environ.get("GEMINI_TIMEOUT_SECONDS")
    os.environ["OPENROUTER_TIMEOUT_SECONDS"] = str(timeout_seconds)
    os.environ["GEMINI_TIMEOUT_SECONDS"] = str(timeout_seconds)
    try:
        event = {
            "event_ticker": "preflight-gpt-smoke",
            "market_ticker": "preflight-gpt-smoke",
            "title": "Will this GPT smoke test return valid probabilities?",
            "description": "Synthetic GPT validation event.",
            "category": "Politics",
            "rules": "Resolves Yes if this synthetic event is marked true.",
            "close_time": "2026-12-31T23:59:59Z",
            "outcomes": ["YES", "NO"],
        }
        reserve_seconds = float(os.environ.get("ARENA_DEADLINE_RESERVE_SECONDS", cfg.arena.deadline_reserve_seconds))
        min_call_seconds = float(os.environ.get("ARENA_MIN_GPT_CALL_SECONDS", cfg.arena.min_gpt_call_seconds))
        smoke_deadline = max(30.0, timeout_seconds + reserve_seconds + min_call_seconds + 10.0)
        forecast = forecast_arena_event(
            event,
            config=cfg,
            use_gpt=True,
            use_live_data=False,
            deadline_seconds=smoke_deadline,
        )
        return {"ok": forecast.source.startswith("gpt"), "source": forecast.source, "audit": forecast.audit}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": _safe_error(exc)}
    finally:
        if old_timeout is None:
            os.environ.pop("OPENROUTER_TIMEOUT_SECONDS", None)
        else:
            os.environ["OPENROUTER_TIMEOUT_SECONDS"] = old_timeout
        if old_gemini_timeout is None:
            os.environ.pop("GEMINI_TIMEOUT_SECONDS", None)
        else:
            os.environ["GEMINI_TIMEOUT_SECONDS"] = old_gemini_timeout


def _safe_get(
    url: str,
    *,
    headers: dict[str, str],
    timeout_seconds: float,
    include_json_keys: tuple[str, ...],
) -> dict[str, Any]:
    try:
        response = requests.get(url, headers=headers, timeout=timeout_seconds)
        payload: dict[str, Any] = {"checked": True, "ok": response.ok, "status_code": response.status_code}
        try:
            data = response.json()
            if include_json_keys:
                payload["safe_json"] = _safe_json_fields(data, include_json_keys)
            else:
                payload["json"] = data
            if not response.ok:
                payload["error"] = _safe_response_error(data)
        except ValueError:
            payload["text_preview"] = response.text[:200]
        return payload
    except Exception as exc:  # noqa: BLE001
        return {"checked": True, "ok": False, "error": _safe_error(exc)}


def _safe_response_error(data: Any) -> str | None:
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error.get("code") or error)[:300]
        if error is not None:
            return str(error)[:300]
    return None


def _safe_json_fields(data: Any, include_json_keys: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    source = data.get("data") if isinstance(data.get("data"), dict) else data
    return {key: source.get(key) for key in include_json_keys if key in source}


def _safe_error(exc: Exception) -> str:
    return f"{type(exc).__name__}:{exc}"[:500]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--spend-gpt", action="store_true")
    args = parser.parse_args()
    print(json.dumps(
        run_preflight(
            offline=args.offline,
            timeout_seconds=args.timeout_seconds,
            spend_gpt=args.spend_gpt,
        ),
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
