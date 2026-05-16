"""Common prompt, parsing, and adapter dispatch for model forecasters."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from prep.schemas import (
    ForecastDiagnostics,
    ForecastValues,
    MarketPacket,
    ModelForecast,
    ReasoningTrack,
    clamp_prob,
)


SYSTEM_PROMPT = """\
You are a prediction-market forecaster. Return only valid JSON matching the
requested schema. Provide an auditable reasoning track: cite evidence,
assumptions, counterarguments, information gaps, and probability adjustments.
Do not include hidden chain-of-thought. Use concise, explicit reasoning that a
supervisor can inspect.
"""


def build_user_prompt(packet: MarketPacket) -> str:
    schema = {
        "forecast": {
            "p_yes": "float 0.01-0.99 — your P(YES); reason about this direction first",
            "p_no": "float 0.01-0.99 — your P(NO), reasoned independently as a fresh frame. Should be close to (1 - p_yes); we average them to remove primary-direction bias",
            "confidence": "float from 0 to 1",
            "uncertainty": "float from 0 to 1",
            "fair_yes_price": "float from 0.01 to 0.99",
            "max_yes_buy_price": "float from 0.01 to 0.99",
            "max_no_buy_price": "float from 0.01 to 0.99",
            "trade_recommendation": "BUY_YES, BUY_NO, BUY_YES_SMALL, BUY_NO_SMALL, or NO_TRADE",
        },
        "reasoning_track": {
            "summary": "short thesis",
            "base_rate": "base-rate reasoning",
            "market_analysis": "how Kalshi price influenced your estimate",
            "key_evidence": [{"claim": "...", "source": "...", "impact": "+0.03 YES"}],
            "counterarguments": [{"claim": "...", "impact": "-0.02 YES"}],
            "assumptions": ["..."],
            "information_gaps": ["..."],
            "what_would_change_my_mind": ["..."],
        },
        "diagnostics": {
            "evidence_quality": "low, medium, or high",
            "rules_clarity": "low, medium, or high",
            "liquidity_quality": "low, medium, or high",
            "market_disagreement_reason": "short string",
            "should_defer_to_market": "boolean",
        },
    }
    return (
        "Forecast this Kalshi binary market. Estimate fair probability, not just "
        "whether to trade.\n\n"
        f"MARKET_PACKET:\n{json.dumps(packet.to_dict(), indent=2, sort_keys=True)}\n\n"
        f"REQUIRED_JSON_SCHEMA:\n{json.dumps(schema, indent=2)}"
    )


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start:end + 1]
    return json.loads(text)


def forecast_from_response(
    *,
    provider: str,
    model_id: str,
    packet: MarketPacket,
    response: dict[str, Any],
    raw_response: dict[str, Any] | None = None,
) -> ModelForecast:
    forecast = response.get("forecast") or {}
    reasoning = response.get("reasoning_track") or {}
    diagnostics = response.get("diagnostics") or {}

    # Bi-direction averaging: if the model returned a separate P(NO), treat it
    # as an independent estimate of (1 - p_yes) and average. Removes the
    # primary-direction bias most LLMs exhibit when asked one-sided questions.
    raw_p_yes = forecast.get("p_yes", packet.kalshi.market_mid)
    raw_p_no = forecast.get("p_no")
    if raw_p_no is not None:
        try:
            p_yes_effective = (float(raw_p_yes) + (1.0 - float(raw_p_no))) / 2.0
        except (TypeError, ValueError):
            p_yes_effective = raw_p_yes
    else:
        p_yes_effective = raw_p_yes

    values = ForecastValues(
        p_yes=p_yes_effective,
        confidence=forecast.get("confidence", 0.5),
        uncertainty=forecast.get("uncertainty", 0.5),
        fair_yes_price=forecast.get("fair_yes_price"),
        max_yes_buy_price=forecast.get("max_yes_buy_price"),
        max_no_buy_price=forecast.get("max_no_buy_price"),
        trade_recommendation=forecast.get("trade_recommendation", "NO_TRADE"),
    )
    return ModelForecast(
        model_id=model_id,
        provider=provider,
        as_of=packet.as_of,
        market_ticker=packet.market_ticker,
        forecast=values,
        reasoning_track=ReasoningTrack(
            summary=str(reasoning.get("summary", "")),
            base_rate=str(reasoning.get("base_rate", "")),
            market_analysis=str(reasoning.get("market_analysis", "")),
            key_evidence=list(reasoning.get("key_evidence") or []),
            counterarguments=list(reasoning.get("counterarguments") or []),
            assumptions=list(reasoning.get("assumptions") or []),
            information_gaps=list(reasoning.get("information_gaps") or []),
            what_would_change_my_mind=list(reasoning.get("what_would_change_my_mind") or []),
        ),
        diagnostics=ForecastDiagnostics(
            evidence_quality=diagnostics.get("evidence_quality", "medium"),
            rules_clarity=diagnostics.get("rules_clarity", "medium"),
            liquidity_quality=diagnostics.get("liquidity_quality", "medium"),
            market_disagreement_reason=str(diagnostics.get("market_disagreement_reason", "")),
            should_defer_to_market=bool(diagnostics.get("should_defer_to_market", True)),
        ),
        raw_response=raw_response or response,
    )


@dataclass
class ForecasterConfig:
    name: str
    provider: str
    model: str
    api_key_env: str | None = None
    enabled: bool = True
    weight: float = 1.0
    temperature: float = 0.1
    max_tokens: int = 1400
    reasoning_effort: str | None = None
    mock_edge_bps: float = 0.0
    # OpenRouter-only: ordered list of fallback model IDs. If the primary model
    # errors or is unavailable, OR auto-routes to the next one in the list.
    # Other providers ignore this.
    fallback_models: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ForecasterConfig":
        return cls(
            name=data["name"],
            provider=data["provider"],
            model=data["model"],
            api_key_env=data.get("api_key_env"),
            enabled=bool(data.get("enabled", True)),
            weight=float(data.get("weight", 1.0)),
            temperature=float(data.get("temperature", 0.1)),
            max_tokens=int(data.get("max_tokens", 1400)),
            reasoning_effort=data.get("reasoning_effort"),
            fallback_models=list(data.get("fallback_models") or []),
            mock_edge_bps=float(data.get("mock_edge_bps", 0.0)),
        )


def stable_prompt_hash(packet: MarketPacket, config: ForecasterConfig) -> str:
    payload = {
        "packet": packet.to_dict(),
        "model": config.model,
        "provider": config.provider,
        "prompt": SYSTEM_PROMPT,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def heuristic_trade_recommendation(p_yes: float, packet: MarketPacket, min_edge: float = 0.06) -> str:
    yes_ask = packet.kalshi.yes_ask
    no_ask = packet.kalshi.no_ask
    yes_edge = -1.0 if yes_ask is None else p_yes - yes_ask
    no_edge = -1.0 if no_ask is None else (1.0 - p_yes) - no_ask
    if yes_edge > min_edge and yes_edge >= no_edge:
        return "BUY_YES" if yes_edge > min_edge * 1.75 else "BUY_YES_SMALL"
    if no_edge > min_edge:
        return "BUY_NO" if no_edge > min_edge * 1.75 else "BUY_NO_SMALL"
    return "NO_TRADE"


def forecast_from_config(config: ForecasterConfig, packet: MarketPacket) -> ModelForecast:
    if config.provider == "mock":
        from .mock import forecast
        return forecast(config, packet)
    if config.provider == "gemini":
        from .gemini import forecast
        return forecast(config, packet)
    if config.provider == "openrouter":
        from .openrouter import forecast
        return forecast(config, packet)
    if config.provider == "xai":
        from .xai import forecast
        return forecast(config, packet)
    raise ValueError(f"Unknown forecaster provider: {config.provider}")


def clamped_market_plus_edge(packet: MarketPacket, edge_bps: float) -> float:
    return clamp_prob(packet.kalshi.market_mid + edge_bps / 10_000.0)
