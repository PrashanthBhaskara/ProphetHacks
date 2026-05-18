"""Adapter for Dhruv's Gemini forecasting lane.

Dhruv's runtime has its own Arena packet, deterministic prior, source/evidence,
deadline, and JSON-repair logic. This adapter keeps that method intact while
converting its output into the shared `ModelForecast` contract consumed by the
ensemble and judge.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from dhruv_gpt_forecasting.arena_agent import ensemble_response_from_forecast, forecast_arena_event
from dhruv_gpt_forecasting.config import load_config

from .base import ForecasterConfig
from prep.schemas import (
    ForecastDiagnostics,
    ForecastValues,
    KalshiQuote,
    MarketPacket,
    ModelForecast,
    ReasoningTrack,
    normalize_distribution,
)


def forecast(config: ForecasterConfig, packet: MarketPacket) -> ModelForecast:
    dhruv_cfg = load_config(_adapter_config_path(config))
    dhruv_cfg.model = replace(
        dhruv_cfg.model,
        name=config.name,
        provider=config.llm_backend or dhruv_cfg.model.provider,
        model=config.model,
        api_key_env=config.api_key_env or dhruv_cfg.model.api_key_env,
        api_key_fallback_envs=list(config.api_key_fallback_envs or dhruv_cfg.model.api_key_fallback_envs),
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        native_search_grounding_enabled=bool(
            config.enable_google_search
            or dhruv_cfg.model.native_search_grounding_enabled
        ),
    )

    event = _event_from_packet(packet)
    arena_forecast = forecast_arena_event(
        event,
        config=dhruv_cfg,
        use_gpt=config.use_gpt,
        use_live_data=config.use_live_data,
        deadline_seconds=config.deadline_seconds,
        external_evidence=_external_evidence(packet),
    )
    return _to_model_forecast(config, packet, arena_forecast)


def _adapter_config_path(config: ForecasterConfig) -> Path | None:
    if not config.adapter_config_path:
        return None
    path = Path(config.adapter_config_path)
    if path.is_absolute():
        return path
    prep_root = Path(__file__).resolve().parents[3]
    for candidate in (Path.cwd() / path, prep_root / path):
        if candidate.exists():
            return candidate
    return path


def _event_from_packet(packet: MarketPacket) -> dict[str, Any]:
    retrieval = dict(packet.retrieval or {})
    features = dict(retrieval.get("features") or {})
    features.update({
        "handoff_retrieval": retrieval,
        "market_implied_probabilities": retrieval.get("market_implied_probabilities"),
    })
    if _has_quote(packet.kalshi) and packet.is_binary:
        mid = packet.kalshi.market_mid
        features["handoff_kalshi_market_mid"] = mid
        features["market_implied_probabilities"] = {"YES": mid, "NO": 1.0 - mid}

    return {
        "event_ticker": packet.event_ticker,
        "market_ticker": packet.market_ticker,
        "task_id": packet.market_ticker or packet.event_ticker,
        "title": packet.title,
        "subtitle": packet.subtitle,
        "description": retrieval.get("description"),
        "context": retrieval.get("context"),
        "category": packet.category,
        "rules": packet.rules,
        "close_time": packet.close_time,
        "predict_by": packet.close_time,
        "as_of": packet.as_of,
        "snapshot_time": packet.as_of,
        "outcomes": list(packet.outcomes or ["YES", "NO"]),
        "features": features,
    }


def _external_evidence(packet: MarketPacket) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    retrieval = dict(packet.retrieval or {})
    market_probs = _clean_probabilities(
        retrieval.get("market_implied_probabilities"),
        packet.outcomes,
    )
    if market_probs:
        evidence.append({
            "source": "handoff_market_implied_probabilities",
            "timestamp": packet.as_of,
            "claim": "Market-implied probabilities were supplied by the handoff packet.",
            "probabilities": market_probs,
        })
    elif _has_quote(packet.kalshi) and packet.is_binary:
        mid = packet.kalshi.market_mid
        evidence.append({
            "source": "handoff_kalshi_quote",
            "timestamp": packet.kalshi.snapshot_time or packet.as_of,
            "claim": "Kalshi quote midpoint from the handoff packet.",
            "probabilities": {"YES": mid, "NO": 1.0 - mid},
            "yes_probability": mid,
        })
    return evidence


def _to_model_forecast(config: ForecasterConfig, packet: MarketPacket, arena_forecast) -> ModelForecast:
    audit = dict(arena_forecast.audit or {})
    fallback_reason = audit.get("fallback_reason")
    authority = audit.get("final_probability_authority")
    should_defer = bool(fallback_reason) or arena_forecast.source == "deterministic_arena_prior"
    should_defer = should_defer or (authority is None and arena_forecast.confidence < 0.45)
    probabilities = (
        _fallback_distribution(packet)
        if fallback_reason
        else dict(arena_forecast.probabilities)
    )

    return ModelForecast(
        model_id=config.model,
        provider=config.provider,
        as_of=packet.as_of,
        market_ticker=packet.market_ticker,
        forecast=ForecastValues(
            probabilities=probabilities,
            confidence=arena_forecast.confidence,
            uncertainty=arena_forecast.uncertainty,
        ),
        reasoning_track=ReasoningTrack(
            summary=_summary(arena_forecast),
            base_rate=_base_rate_note(audit),
            market_analysis=_market_note(arena_forecast, audit),
            context_market_analysis=_context_note(audit),
            key_evidence=list(arena_forecast.key_evidence or []),
            source_audit=_source_audit(arena_forecast),
            counterarguments=list(arena_forecast.counterarguments or []),
            assumptions=_assumptions(arena_forecast, audit),
            information_gaps=list(arena_forecast.information_gaps or []),
            what_would_change_my_mind=[],
        ),
        diagnostics=ForecastDiagnostics(
            evidence_quality=_quality(arena_forecast.confidence),
            rules_clarity="high" if packet.rules else "medium",
            liquidity_quality="medium",
            market_disagreement_reason=str(arena_forecast.calibration_note or ""),
            should_defer_to_market=should_defer,
        ),
        raw_response={
            "dhruv_arena_forecast": arena_forecast.to_dict(),
            "dhruv_lane_envelope": ensemble_response_from_forecast(
                arena_forecast,
                mode="handoff_ensemble_lane",
            ),
        },
    )


def _fallback_distribution(packet: MarketPacket) -> dict[str, float]:
    """Market/default distribution for Dhruv runs that never reached a model.

    If the Dhruv lane falls back because the model key is missing, the model
    call fails, or the lane runs out of budget, it should not act like a
    separate opinionated forecast. It should mirror current market-implied
    probabilities when supplied by the main packet, otherwise use uniform.
    """
    outcomes = packet.outcomes or ["YES", "NO"]
    market_probs = _clean_probabilities(
        (packet.retrieval or {}).get("market_implied_probabilities"),
        outcomes,
    )
    if market_probs:
        return normalize_distribution(market_probs)
    if _has_quote(packet.kalshi) and packet.is_binary:
        mid = packet.kalshi.market_mid
        return {"YES": mid, "NO": 1.0 - mid}
    share = 1.0 / max(1, len(outcomes))
    return {outcome: share for outcome in outcomes}


def _has_quote(quote: KalshiQuote | None) -> bool:
    if quote is None:
        return False
    return any(
        value is not None
        for value in (quote.yes_bid, quote.yes_ask, quote.no_bid, quote.no_ask, quote.last_price)
    )


def _clean_probabilities(raw: Any, outcomes: list[str]) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    case_map = {str(key).casefold(): key for key in raw}
    for outcome in outcomes or ["YES", "NO"]:
        key = outcome if outcome in raw else case_map.get(outcome.casefold())
        if key is None:
            continue
        try:
            value = float(raw[key])
        except (TypeError, ValueError):
            continue
        if value > 1.0:
            value /= 100.0
        out[outcome] = value
    return out


def _summary(arena_forecast) -> str:
    if arena_forecast.calibration_note:
        return str(arena_forecast.calibration_note)
    if arena_forecast.reason_codes:
        return "Dhruv Gemini lane used: " + ", ".join(arena_forecast.reason_codes[:6])
    return f"Dhruv Gemini lane forecast from {arena_forecast.source}."


def _base_rate_note(audit: dict[str, Any]) -> str:
    prior = audit.get("deterministic_prior")
    if isinstance(prior, dict):
        source = prior.get("source")
        reasons = prior.get("reason_codes") or []
        if source or reasons:
            return f"Deterministic prior {source or ''} used {', '.join(map(str, reasons[:6]))}.".strip()
    return "Dhruv lane builds a deterministic prior from the live packet, market context, and grounded evidence."


def _market_note(arena_forecast, audit: dict[str, Any]) -> str:
    fallback_reason = audit.get("fallback_reason")
    if fallback_reason:
        return f"Fallback reason: {fallback_reason}."
    authority = audit.get("final_probability_authority")
    if authority:
        return f"Final probability authority: {authority}."
    return f"Forecast source: {arena_forecast.source}."


def _context_note(audit: dict[str, Any]) -> str:
    sources = audit.get("live_evidence_sources")
    if isinstance(sources, dict) and sources:
        pairs = ", ".join(f"{key}:{value}" for key, value in sorted(sources.items())[:8])
        return f"Live/context evidence sources: {pairs}."
    count = audit.get("live_evidence_count")
    if count:
        return f"Dhruv lane attached {count} live/context evidence items."
    return ""


def _source_audit(arena_forecast) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    audit = dict(arena_forecast.audit or {})
    for item in audit.get("api_logs") or []:
        if isinstance(item, dict):
            rows.append({
                "source": item.get("provider") or "llm_api",
                "source_timestamp": None,
                "cutoff_check": "Provider call was made inside the Dhruv lane runtime.",
                "used": True,
                "reason": f"model={item.get('model')}",
            })
    for item in audit.get("live_evidence_preview") or []:
        if isinstance(item, dict):
            rows.append({
                "source": item.get("source") or "live_evidence",
                "source_timestamp": item.get("timestamp"),
                "cutoff_check": "Included by Dhruv lane evidence assembly.",
                "used": True,
                "reason": str(item.get("claim") or "")[:120],
            })
    return rows[:8]


def _assumptions(arena_forecast, audit: dict[str, Any]) -> list[str]:
    assumptions = [f"Dhruv lane source={arena_forecast.source}."]
    if audit.get("prior_shrink_weight") is not None:
        assumptions.append(f"Prior shrink weight={audit.get('prior_shrink_weight')}.")
    if audit.get("native_search_grounding_enabled") is not None:
        assumptions.append(f"Native search grounding={audit.get('native_search_grounding_enabled')}.")
    return assumptions[:3]


def _quality(confidence: float) -> str:
    if confidence >= 0.65:
        return "high"
    if confidence >= 0.40:
        return "medium"
    return "low"
