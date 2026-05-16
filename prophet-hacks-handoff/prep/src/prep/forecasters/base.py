"""Common prompt, parsing, and adapter dispatch for model forecasters."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
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
    """Build the user prompt asking for a per-outcome probability distribution.

    Per Prophet Arena dev docs, predictions are a distribution over the
    event's `outcomes` list (binary events use ["YES", "NO"]). Probabilities
    don't have to sum to 1 — the scorer normalizes — but we ask for a coherent
    distribution to keep reasoning interpretable.

    For BINARY events we additionally apply bi-direction prompting: ask the
    model to reason about P(YES) and P(NO) as two INDEPENDENT questions and
    intentionally NOT enforce sum-to-1. `forecast_from_response` then averages
    `(p_yes + (1 - p_no)) / 2` to remove the well-documented primary-direction
    bias most LLMs exhibit on binary forecasts. Per PAPER_NOTES.md +
    STRATEGY_FINDINGS.md this is "a free calibration win" on 4/5 models.
    """
    probabilities_schema = {
        outcome: "float 0.01-0.99 — your P(this outcome)"
        for outcome in packet.outcomes
    }
    schema = {
        "forecast": {
            "probabilities": probabilities_schema,
            "confidence": "float 0-1 — how confident you are in this distribution overall",
            "uncertainty": "float 0-1 — residual uncertainty after your reasoning",
            "trade_recommendation": "BUY_YES, BUY_NO, BUY_YES_SMALL, BUY_NO_SMALL, or NO_TRADE (binary-market trading only)",
        },
        "reasoning_track": {
            "summary": "short thesis covering the whole distribution",
            "base_rate": "base-rate reasoning",
            "market_analysis": "how Kalshi price (if available) influenced your estimate",
            "key_evidence": [{"claim": "...", "source": "...", "impact": "+0.03 to <outcome>"}],
            "counterarguments": [{"claim": "...", "impact": "-0.02 to <outcome>"}],
            "assumptions": ["..."],
            "information_gaps": ["..."],
            "what_would_change_my_mind": ["..."],
        },
        "diagnostics": {
            "evidence_quality": "low, medium, or high",
            "rules_clarity": "low, medium, or high",
            "liquidity_quality": "low, medium, or high",
            "market_disagreement_reason": "short string",
            "should_defer_to_market": "boolean (binary markets only)",
        },
    }
    outcome_list = ", ".join(repr(o) for o in packet.outcomes)
    instructions = (
        f"Forecast this market. The event has {len(packet.outcomes)} possible outcomes: {outcome_list}. "
        "Estimate the probability of each outcome. For mutually-exclusive outcomes the probabilities "
        "should be coherent (roughly sum to 1.0). For non-exclusive outcomes (rare) treat each as an "
        "independent binary."
    )
    if packet.is_binary:
        instructions += (
            "\n\nBI-DIRECTION CALIBRATION (binary case): reason about P(YES) and P(NO) as TWO "
            "INDEPENDENT questions. Do not constrain them to sum to 1.0 in your reasoning. "
            "Set p_yes from your strongest case for YES; set p_no from your strongest case for NO. "
            "If you're well-calibrated they'll roughly add to 1; if they don't, the supervisor "
            "averages them to remove your primary-direction bias."
        )
    return (
        instructions
        + f"\n\nMARKET_PACKET:\n{json.dumps(packet.to_dict(), indent=2, sort_keys=True)}\n\n"
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

    # Multi-outcome distribution is canonical. Accept three input forms in
    # priority order: (1) `probabilities` dict, (2) legacy `p_yes` only, (3)
    # market-mid fallback for binary if model returned nothing usable.
    raw_probs = forecast.get("probabilities")
    cleaned_probs: dict[str, float] = {}
    if isinstance(raw_probs, dict):
        for k, v in raw_probs.items():
            try:
                cleaned_probs[str(k)] = float(v)
            except (TypeError, ValueError):
                continue

    # Bi-direction averaging for binary events: if the model returned both
    # YES and NO probabilities that don't sum to ~1, treat NO as an
    # independent estimate of (1 - YES) and average. This is the calibration
    # win from PAPER_NOTES.md. Must be done BEFORE ForecastValues's
    # normalize_distribution would otherwise renormalize the signal away.
    if packet.is_binary and {"YES", "NO"}.issubset(cleaned_probs):
        py_raw = cleaned_probs["YES"]
        pn_raw = cleaned_probs["NO"]
        # Only average when the two estimates disagree by > 5pp; below that
        # the model is already coherent and averaging is a no-op
        if abs((py_raw + pn_raw) - 1.0) > 0.05:
            py_eff = (py_raw + (1.0 - pn_raw)) / 2.0
            py_eff = max(0.01, min(0.99, py_eff))
            cleaned_probs = {"YES": py_eff, "NO": 1.0 - py_eff}

    if not cleaned_probs:
        # Legacy binary path: model returned `p_yes` only. Build a 2-outcome
        # distribution against the packet's outcomes (which should be ["YES","NO"]).
        p_yes_raw = forecast.get("p_yes")
        if p_yes_raw is None:
            p_yes_raw = packet.kalshi.market_mid
        try:
            py = float(p_yes_raw)
        except (TypeError, ValueError):
            py = 0.5
        # Use packet.outcomes labels; default ["YES","NO"] if multi-outcome
        # event got a one-direction response (suboptimal but won't crash).
        if len(packet.outcomes) >= 2:
            cleaned_probs = {packet.outcomes[0]: py, packet.outcomes[1]: 1.0 - py}
        else:
            cleaned_probs = {(packet.outcomes[0] if packet.outcomes else "YES"): py}

    values = ForecastValues(
        probabilities=cleaned_probs,
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
    raise ValueError(f"Unknown forecaster provider: {config.provider}")


def clamped_market_plus_edge(packet: MarketPacket, edge_bps: float) -> float:
    return clamp_prob(packet.kalshi.market_mid + edge_bps / 10_000.0)
