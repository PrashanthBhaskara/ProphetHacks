"""Prophet Arena Brier-optimized local agent entrypoint."""

from __future__ import annotations

import json
import hashlib
import math
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .arena_live_data import gather_live_evidence
from .arena_priors import build_arena_packet, deterministic_arena_prior
from .arena_types import ArenaForecast, ArenaForecastPacket, ArenaPrior
from .config import ForecastConfig, load_config, load_local_env, resolve_api_key
from .constraints import enforce_constraints, normalize_distribution
from .evidence_sources import annotate_evidence_items, canonical_source_name, evidence_source_policy
from .features import parse_dt
from .grounded_research import gather_grounded_research_evidence
from .openrouter import call_openrouter_json
from .prompts import arena_audit_messages, arena_messages, arena_repair_messages, prompt_hash


def forecast_arena_event(
    event: dict[str, Any],
    *,
    config: ForecastConfig | None = None,
    use_gpt: bool | None = None,
    use_live_data: bool | None = None,
    deadline_seconds: float | None = None,
) -> ArenaForecast:
    """Return a Brier-only forecast over the event's exact outcome labels."""
    started = time.monotonic()
    load_local_env()
    cfg = config or load_config()
    deadline = _deadline_seconds(deadline_seconds, cfg)
    packet = build_arena_packet(event)
    if use_gpt is None:
        use_gpt = _gpt_enabled(cfg)
    live_data_enabled = _live_data_enabled(use_live_data, cfg)
    evidence_deadline_at = _evidence_deadline_at(started, deadline, cfg)
    grounded_research: list[dict[str, Any]] = []
    if live_data_enabled and use_gpt and _can_continue(evidence_deadline_at):
        # Run the generative source-reading pass before slower vendor fetches so
        # native search grounding is prioritized inside the live evidence budget.
        grounded_research = gather_grounded_research_evidence(
            packet,
            cfg,
            deadline_at=evidence_deadline_at,
            existing_evidence=[],
        )
    live_evidence = gather_live_evidence(
        packet,
        cfg,
        enabled=live_data_enabled,
        deadline_at=evidence_deadline_at,
        allow_llm_queries=bool(use_gpt),
    )
    live_evidence = [*grounded_research, *live_evidence]
    packet.live_evidence = annotate_evidence_items(live_evidence[: cfg.arena.max_live_evidence], packet.category)
    observed_sources = [str(item.get("source") or "") for item in packet.live_evidence]
    packet.features["evidence_source_policy"] = evidence_source_policy(packet.category, observed_sources)
    packet.features["gpt_final_probability_model"] = True
    prior = deterministic_arena_prior(packet)
    api_logs: list[dict[str, Any]] = []
    errors: list[str] = []

    if not use_gpt:
        final = _forecast_from_prior(packet, prior, audit={"mode": "deterministic_only"})
        _attach_deadline_audit(final, started, deadline)
        return final
    if not _has_call_budget(started, deadline, cfg):
        final = _forecast_from_prior(
            packet,
            prior,
            audit={
                "mode": "deterministic_fallback",
                "fallback_reason": "deadline_budget_before_primary_gpt",
                "errors": ["deadline_budget_before_primary_gpt"],
            },
        )
        _attach_deadline_audit(final, started, deadline)
        return final

    model = cfg.cheap_model
    if resolve_api_key(model)[0] is None:
        final = _forecast_from_prior(
            packet,
            prior,
            audit={
                "mode": "deterministic_fallback",
                "fallback_reason": "missing_api_key",
                "errors": [f"missing_api_key:{model.api_key_env}"],
            },
        )
        _attach_deadline_audit(final, started, deadline)
        return final

    primary_messages = arena_messages(packet, prior, model_id=model.model)
    search_grounding_enabled = _search_grounding_enabled(packet, cfg)
    forecast_cache_path = _forecast_cache_path(
        cfg,
        event,
        packet,
        model.model,
        prompt_hash(primary_messages, model.model),
        use_live_data=live_data_enabled,
        search_grounding_enabled=search_grounding_enabled,
    )
    cached_forecast = _read_cached_forecast(forecast_cache_path)
    if cached_forecast is not None:
        cached_forecast.audit["forecast_cache_hit"] = True
        _attach_deadline_audit(cached_forecast, started, deadline)
        return cached_forecast

    first = None
    remaining_at_gpt_start = _remaining_seconds(started, deadline)
    try:
        payload, call_log = _cached_json_call(
            cfg,
            messages=primary_messages,
            cache_namespace="arena_primary",
            started=started,
            deadline_seconds=deadline,
            search_grounding=search_grounding_enabled,
        )
        api_logs.append(call_log)
        first = arena_forecast_from_payload(payload, packet, source="gpt_primary")
    except Exception as exc:  # noqa: BLE001 - deterministic fallback is required.
        errors.append(f"primary:{type(exc).__name__}:{exc}")
        try:
            if not _has_call_budget(started, deadline, cfg):
                raise TimeoutError("deadline_budget_before_repair_gpt")
            payload, call_log = _cached_json_call(
                cfg,
                messages=arena_repair_messages(packet, prior, str(exc), model_id=model.model),
                cache_namespace="arena_repair",
                started=started,
                deadline_seconds=deadline,
                search_grounding=search_grounding_enabled,
            )
            api_logs.append(call_log)
            first = arena_forecast_from_payload(payload, packet, source="gpt_repair")
        except Exception as repair_exc:  # noqa: BLE001
            errors.append(f"repair:{type(repair_exc).__name__}:{repair_exc}")

    if first is None:
        final = _forecast_from_prior(
            packet,
            prior,
            audit={
                "mode": "deterministic_fallback",
                "fallback_reason": "gpt_failed",
                "api_logs": api_logs,
                "errors": errors,
                "remaining_seconds_at_gpt_start": remaining_at_gpt_start,
            },
        )
        _attach_deadline_audit(final, started, deadline)
        return final

    final = _blend_with_prior(packet, prior, first, cfg)
    if (
        cfg.arena.second_pass_enabled
        and _should_second_pass(packet, prior, final, cfg)
        and _has_call_budget(started, deadline, cfg)
    ):
        try:
            payload, call_log = _cached_json_call(
                cfg,
                messages=arena_audit_messages(packet, prior, final, model_id=model.model),
                cache_namespace="arena_audit",
                started=started,
                deadline_seconds=deadline,
                search_grounding=search_grounding_enabled,
            )
            api_logs.append(call_log)
            audited = arena_forecast_from_payload(payload, packet, source="gpt_audit")
            final = _blend_with_prior(packet, prior, audited, cfg)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"audit:{type(exc).__name__}:{exc}")

    final.audit.update({
        "mode": "arena_forecast_mode",
        "model": model.model,
        "api_logs": api_logs,
        "errors": errors,
        "fallback_reason": None,
        "remaining_seconds_at_gpt_start": remaining_at_gpt_start,
        "deterministic_prior": prior.to_dict(),
        "native_search_grounding_enabled": search_grounding_enabled,
        "search_grounding_engine": model.search_grounding_engine if search_grounding_enabled else None,
        "live_evidence_count": len(packet.live_evidence),
        "live_evidence_sources": _live_evidence_sources(packet.live_evidence),
        "live_evidence_preview": _live_evidence_preview(packet.live_evidence),
        "live_evidence_errors": _live_evidence_errors(packet.live_evidence),
    })
    _attach_deadline_audit(final, started, deadline)
    _write_cached_forecast(forecast_cache_path, final)
    return final


def predict(event: dict[str, Any]) -> dict[str, Any]:
    """Prophet CLI entrypoint: predict(event) -> {"probabilities": [...]}."""
    return forecast_arena_event(event).to_prediction_response()


def arena_forecast_from_payload(
    payload: dict[str, Any],
    packet: ArenaForecastPacket,
    *,
    source: str,
) -> ArenaForecast:
    raw_probs = payload.get("probabilities")
    if not isinstance(raw_probs, dict):
        raw_probs = {}
    probs = enforce_constraints(
        {str(key): float(value) for key, value in raw_probs.items() if _is_number(value)},
        packet.outcomes,
        packet.event_structure,
        lo=0.001,
        hi=0.999,
    )
    return ArenaForecast(
        probabilities=probs,
        confidence=_bounded(payload.get("confidence"), 0.50),
        uncertainty=_bounded(payload.get("uncertainty"), 0.50),
        source=source,
        reason_codes=[str(item) for item in payload.get("reason_codes") or []][:12],
        key_evidence=_list_of_dicts(payload.get("key_evidence"))[:8],
        counterarguments=_list_of_dicts(payload.get("counterarguments"))[:8],
        information_gaps=[str(item) for item in payload.get("information_gaps") or []][:8],
        calibration_note=str(payload.get("calibration_note") or "")[:500] or None,
    )


def _forecast_from_prior(
    packet: ArenaForecastPacket,
    prior: ArenaPrior,
    *,
    audit: dict[str, Any] | None = None,
) -> ArenaForecast:
    return ArenaForecast(
        probabilities=prior.probabilities,
        confidence=prior.confidence,
        uncertainty=prior.uncertainty,
        source=prior.source,
        reason_codes=prior.reason_codes,
        key_evidence=[{
            "claim": "Deterministic historical/base-rate prior used as final forecast.",
            "source": prior.source,
            "impact": "fallback forecast",
        }],
        counterarguments=[],
        information_gaps=["No valid GPT forecast was available."] if audit and audit.get("errors") else [],
        calibration_note="Forecast uses deterministic priors and exact-label normalization.",
        audit=audit or {},
    )


def _blend_with_prior(
    packet: ArenaForecastPacket,
    prior: ArenaPrior,
    forecast: ArenaForecast,
    cfg: ForecastConfig,
) -> ArenaForecast:
    shrink = max(0.0, min(0.75, cfg.arena.prior_shrink_weight))
    shrink = max(shrink, _dynamic_prior_shrink(packet, forecast))
    if shrink <= 0.0:
        forecast.probabilities = normalize_distribution(
            forecast.probabilities,
            packet.outcomes,
            lo=cfg.arena.probability_floor,
            hi=cfg.arena.probability_ceiling,
        )
        forecast.audit["prior_shrink_weight"] = 0.0
        forecast.audit["final_probability_authority"] = "gpt"
        return forecast
    if forecast.uncertainty > 0.70 or forecast.confidence < 0.35:
        shrink = max(shrink, 0.35)
    raw = {
        outcome: (1.0 - shrink) * forecast.probabilities.get(outcome, 0.0)
        + shrink * prior.probabilities.get(outcome, 0.0)
        for outcome in packet.outcomes
    }
    probs = normalize_distribution(
        raw,
        packet.outcomes,
        lo=cfg.arena.probability_floor,
        hi=cfg.arena.probability_ceiling,
    )
    forecast.probabilities = probs
    forecast.audit["prior_shrink_weight"] = shrink
    forecast.audit["final_probability_authority"] = "gpt_with_calibration_shrink"
    return forecast


def _should_second_pass(
    packet: ArenaForecastPacket,
    prior: ArenaPrior,
    forecast: ArenaForecast,
    cfg: ForecastConfig,
) -> bool:
    if forecast.confidence <= cfg.arena.second_pass_low_confidence:
        return True
    if _max_delta(prior.probabilities, forecast.probabilities, packet.outcomes) >= cfg.arena.second_pass_delta_pp:
        return True
    if len(packet.outcomes) > 2 and _entropy(forecast.probabilities) >= cfg.arena.second_pass_high_entropy:
        return True
    if any(item.get("source") == "live_fetch_error" for item in packet.live_evidence):
        return True
    return False


def _cached_json_call(
    cfg: ForecastConfig,
    *,
    messages: list[dict[str, str]],
    cache_namespace: str,
    started: float | None = None,
    deadline_seconds: float | None = None,
    search_grounding: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model = cfg.cheap_model
    p_hash = prompt_hash(messages, model.model)
    cache_dir = Path(cfg.budget.log_dir) / "llm_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    grounding_part = "grounded" if search_grounding else "ungrounded"
    cache_path = cache_dir / f"{cache_namespace}_{model.model.replace('/', '_')}_{grounding_part}_{p_hash}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return cached["payload"], {"cache_hit": True, "prompt_hash": p_hash, "model": model.model}
    payload, call_log = call_openrouter_json(
        model=model,
        messages=messages,
        budget=cfg.budget,
        cache_key=cache_namespace,
        timeout_seconds=_llm_timeout_seconds(cfg, started, deadline_seconds),
        search_grounding=search_grounding,
    )
    cache_path.write_text(
        json.dumps({"payload": payload, "call_log": call_log.to_dict()}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload, call_log.to_dict()


def _gpt_enabled(cfg: ForecastConfig) -> bool:
    if _env_bool("ARENA_OFFLINE", False) or _env_bool("ARENA_DISABLE_GPT", False):
        return False
    return _env_bool("ARENA_ENABLE_GPT", cfg.arena.gpt_enabled_default)


def _live_data_enabled(value: bool | None, cfg: ForecastConfig) -> bool:
    if value is not None:
        enabled = bool(value)
    else:
        enabled = _env_bool("ARENA_ENABLE_LIVE_DATA", cfg.arena.live_data_enabled_default)
    return enabled and not _env_bool("ARENA_DISABLE_LIVE_DATA", False)


def _deadline_seconds(value: float | None, cfg: ForecastConfig) -> float | None:
    if value is not None:
        return float(value)
    raw = os.environ.get("ARENA_RESPONSE_DEADLINE_SECONDS")
    if not raw:
        return cfg.arena.response_deadline_seconds
    try:
        return float(raw)
    except ValueError:
        return cfg.arena.response_deadline_seconds


def _has_call_budget(started: float, deadline_seconds: float | None, cfg: ForecastConfig) -> bool:
    if deadline_seconds is None:
        return True
    reserve = float(os.environ.get("ARENA_DEADLINE_RESERVE_SECONDS", cfg.arena.deadline_reserve_seconds))
    min_call = float(os.environ.get("ARENA_MIN_GPT_CALL_SECONDS", cfg.arena.min_gpt_call_seconds))
    return (time.monotonic() - started) + reserve + min_call < deadline_seconds


def _attach_deadline_audit(forecast: ArenaForecast, started: float, deadline_seconds: float | None) -> None:
    elapsed = time.monotonic() - started
    forecast.audit["elapsed_seconds"] = elapsed
    forecast.audit["deadline_seconds"] = deadline_seconds
    forecast.audit["within_deadline"] = True if deadline_seconds is None else elapsed <= deadline_seconds
    forecast.audit["response_deadline_seconds"] = deadline_seconds
    forecast.audit["within_response_deadline"] = True if deadline_seconds is None else elapsed <= deadline_seconds


def _remaining_seconds(started: float, deadline_seconds: float | None) -> float | None:
    if deadline_seconds is None:
        return None
    return max(0.0, deadline_seconds - (time.monotonic() - started))


def _evidence_deadline_at(started: float, deadline_seconds: float | None, cfg: ForecastConfig) -> float:
    evidence_budget = float(os.environ.get(
        "ARENA_TOTAL_EVIDENCE_TIMEOUT_SECONDS",
        cfg.arena.total_evidence_timeout_seconds,
    ))
    deadline_at = started + max(0.0, evidence_budget)
    if deadline_seconds is not None:
        reserve = float(os.environ.get("ARENA_DEADLINE_RESERVE_SECONDS", cfg.arena.deadline_reserve_seconds))
        deadline_at = min(deadline_at, started + max(0.0, deadline_seconds - reserve))
    return deadline_at


def _llm_timeout_seconds(
    cfg: ForecastConfig,
    started: float | None,
    deadline_seconds: float | None,
) -> float:
    configured = float(os.environ.get(
        "GEMINI_TIMEOUT_SECONDS",
        os.environ.get("OPENROUTER_TIMEOUT_SECONDS", cfg.arena.llm_timeout_seconds),
    ))
    if started is None or deadline_seconds is None:
        return configured
    reserve = float(os.environ.get("ARENA_DEADLINE_RESERVE_SECONDS", cfg.arena.deadline_reserve_seconds))
    remaining = deadline_seconds - (time.monotonic() - started) - reserve
    return max(0.1, min(configured, remaining))


def _dynamic_prior_shrink(packet: ArenaForecastPacket, forecast: ArenaForecast) -> float:
    evidence_scores = [
        float(item.get("retrieval_confidence", {}).get("overall"))
        for item in packet.live_evidence
        if isinstance(item.get("retrieval_confidence"), dict)
        and _is_number(item.get("retrieval_confidence", {}).get("overall"))
    ]
    evidence_confidence = sum(evidence_scores) / len(evidence_scores) if evidence_scores else 0.0
    shrink = 0.0
    if evidence_confidence <= 0.0:
        shrink = 0.20
    elif evidence_confidence < 0.35:
        shrink = 0.30
    elif evidence_confidence < 0.55:
        shrink = 0.18
    if forecast.uncertainty > 0.70 or forecast.confidence < 0.35:
        shrink = max(shrink, 0.35)
    return max(0.0, min(0.75, shrink))


def _forecast_cache_path(
    cfg: ForecastConfig,
    event: dict[str, Any],
    packet: ArenaForecastPacket,
    model_id: str,
    p_hash: str,
    *,
    use_live_data: bool,
    search_grounding_enabled: bool,
) -> Path:
    manifest_ids = packet.features.get("evidence_manifest_ids") or event.get("evidence_manifest_ids") or []
    cache_key = hashlib.sha256(json.dumps(
        {
            "event": event,
            "as_of": packet.as_of,
            "market_ticker": packet.market_ticker,
            "model": model_id,
            "prompt_hash": p_hash,
            "manifest_ids": manifest_ids,
            "use_live_data": use_live_data,
            "native_search_grounding_enabled": search_grounding_enabled,
            "prior_shrink_weight": cfg.arena.prior_shrink_weight,
        },
        sort_keys=True,
        default=str,
    ).encode("utf-8")).hexdigest()
    return Path(cfg.budget.log_dir) / "forecast_cache" / f"{cache_key}.json"


def _read_cached_forecast(path: Path) -> ArenaForecast | None:
    if not _env_bool("ARENA_ENABLE_FORECAST_CACHE", True) or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        forecast_data = dict(data["forecast"])
        forecast_data.pop("prediction_response", None)
        return ArenaForecast(**forecast_data)
    except Exception:  # noqa: BLE001 - bad cache entries should not block forecasts.
        return None


def _write_cached_forecast(path: Path, forecast: ArenaForecast) -> None:
    if not _env_bool("ARENA_ENABLE_FORECAST_CACHE", True):
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"forecast": forecast.to_dict()}, sort_keys=True) + "\n", encoding="utf-8")
    except Exception:  # noqa: BLE001 - cache write failures are non-fatal.
        return


def _bounded(value: Any, default: float) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        x = default
    return max(0.0, min(1.0, x))


def _is_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _max_delta(left: dict[str, float], right: dict[str, float], outcomes: list[str]) -> float:
    return max(abs(left.get(outcome, 0.0) - right.get(outcome, 0.0)) for outcome in outcomes) if outcomes else 0.0


def _entropy(probs: dict[str, float]) -> float:
    values = [p for p in probs.values() if p > 0.0]
    if len(values) <= 1:
        return 0.0
    entropy = -sum(p * math.log(p) for p in values)
    return entropy / math.log(len(values))


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _can_continue(deadline_at: float | None) -> bool:
    return deadline_at is None or time.monotonic() < deadline_at


def _search_grounding_enabled(packet: ArenaForecastPacket, cfg: ForecastConfig) -> bool:
    model = cfg.cheap_model
    env_value = os.environ.get("GEMINI_NATIVE_SEARCH_GROUNDING", os.environ.get("OPENROUTER_NATIVE_SEARCH_GROUNDING"))
    if env_value is not None:
        enabled = env_value.strip().lower() in {"1", "true", "yes", "on"}
    else:
        enabled = model.native_search_grounding_enabled
    if not enabled:
        return False
    if not model.native_search_grounding_live_only:
        return True
    as_of_dt = parse_dt(packet.as_of)
    if as_of_dt is None:
        return False
    now = datetime.now(timezone.utc)
    max_age = max(0, cfg.arena.pit_external_max_live_age_minutes) * 60
    return abs((now - as_of_dt).total_seconds()) <= max_age


def _live_evidence_sources(items: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(canonical_source_name(item.get("source")) for item in items if isinstance(item, dict))
    return dict(sorted(counts.items()))


def _live_evidence_preview(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    preview: list[dict[str, Any]] = []
    for item in items[:8]:
        if not isinstance(item, dict):
            continue
        row = {
            "source": item.get("source"),
            "claim": item.get("claim"),
            "record_count": item.get("record_count"),
            "source_counts": item.get("source_counts"),
            "retrieval_confidence": item.get("retrieval_confidence"),
        }
        preview.append({key: value for key, value in row.items() if value is not None})
    return preview


def _live_evidence_errors(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        source = item.get("source")
        error = item.get("error")
        claim = item.get("claim")
        if error or source in {"live_fetch_error", "pit_external_fetch_error"}:
            row = {
                "source": source,
                "claim": claim,
                "error": error,
            }
            errors.append({key: value for key, value in row.items() if value is not None})
    return errors[:12]
