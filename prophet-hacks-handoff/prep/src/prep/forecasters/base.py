"""Common prompt, parsing, and adapter dispatch for model forecasters."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
)
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prep.schemas import (
    ForecastDiagnostics,
    ForecastValues,
    MarketPacket,
    ModelForecast,
    ReasoningTrack,
    clamp_prob,
    is_yes_no_outcomes,
    normalize_distribution,
)

logger = logging.getLogger(__name__)

# Per-lane wall-clock budget. Every dispatched forecaster gets at most this
# long before we drop a market-mirror placeholder into the ensemble. The
# default is 7.5 minutes so the judge and API response fit under the 10-minute
# Prophet Arena ceiling. Override with LANE_TIMEOUT_SECONDS.
DEFAULT_LANE_TIMEOUT_SECONDS = 450.0


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
    """
    probabilities_schema = {
        outcome: "float 0.01-0.99 — your P(this outcome)"
        for outcome in packet.outcomes
    }
    schema = {
        "forecast": {
            "prior_probabilities": {
                outcome: "float 0.01-0.99 — market/base-rate prior before your evidence adjustment"
                for outcome in packet.outcomes
            },
            "probability_adjustments": [
                {
                    "outcome": "one listed outcome or ALL",
                    "delta": "signed float adjustment from prior to final probability",
                    "reason": "one short source-backed reason",
                },
            ],
            "probabilities": probabilities_schema,
            "confidence": "float 0-1 — how confident you are in this distribution overall",
            "uncertainty": "float 0-1 — residual uncertainty after your reasoning",
        },
        "reasoning_track": {
            "summary": "short thesis covering the whole distribution",
            "base_rate": "base-rate reasoning",
            "market_analysis": "how Kalshi price (if available) influenced your estimate",
            "context_market_analysis": "how related/sibling markets influenced your estimate, if provided",
            "key_evidence": [
                {
                    "claim": "concise claim, <=160 chars",
                    "source": "...",
                    "source_type": "packet, context_market, official_primary, reputable_reporting, search_result, etc.",
                    "source_timestamp": "ISO timestamp or date strictly before MARKET_PACKET.as_of",
                    "impact": "+0.03 to <outcome>",
                },
            ],
            "source_audit": [
                {
                    "source": "...",
                    "source_timestamp": "ISO timestamp or date strictly before MARKET_PACKET.as_of",
                    "cutoff_check": "concise proof source was observable before MARKET_PACKET.as_of",
                    "used": "boolean",
                    "reason": "why used or excluded, <=120 chars",
                },
            ],
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
            "should_defer_to_market": "boolean — true when market-implied or base-rate priors should dominate",
        },
    }
    outcome_list = ", ".join(repr(o) for o in packet.outcomes)
    return (
        f"Forecast this market. The event has {len(packet.outcomes)} possible outcomes: {outcome_list}. "
        "Estimate the probability of each outcome. First set prior_probabilities from market_implied_probabilities "
        "when available, otherwise from base rates and related-market constraints. Then make small, explicit "
        "probability_adjustments only when timestamp-valid evidence justifies moving away from the prior. "
        "For mutually-exclusive outcomes the final probabilities "
        "should be coherent (roughly sum to 1.0). For non-exclusive outcomes (rare) treat each as an "
        "independent binary.\n\n"
        f"MARKET_PACKET:\n{json.dumps(packet.to_dict(), indent=2, sort_keys=True)}\n\n"
        "OUTPUT_BUDGET:\n"
        "- Keep reasoning concise so the full JSON completes.\n"
        "- key_evidence: at most 5 items.\n"
        "- source_audit: at most 8 items.\n"
        "- counterarguments, assumptions, information_gaps, what_would_change_my_mind: at most 3 items each.\n"
        "- Each free-text field should be one short sentence.\n\n"
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
        is_mutually_exclusive=packet.is_mutually_exclusive,
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
            context_market_analysis=str(reasoning.get("context_market_analysis", "")),
            key_evidence=list(reasoning.get("key_evidence") or []),
            source_audit=list(reasoning.get("source_audit") or []),
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


def resolve_api_key(config: "ForecasterConfig", default_env: str) -> str | None:
    """Return the first non-empty API key found across primary and fallback envs."""
    for env_name in [config.api_key_env or default_env, *config.api_key_fallback_envs]:
        value = os.environ.get(env_name)
        if value:
            return value
    return None


@dataclass
class ForecasterConfig:
    name: str
    provider: str
    model: str
    api_key_env: str | None = None
    api_key_fallback_envs: list[str] = field(default_factory=list)
    enabled: bool = True
    weight: float = 1.0
    temperature: float = 0.1
    max_tokens: int = 1400
    reasoning_effort: str | None = None
    system_prompt: str | None = None
    system_prompt_path: str | None = None
    enable_google_search: bool = True
    require_google_search_grounding: bool = False
    adapter_config_path: str | None = None
    deadline_seconds: float | None = None
    use_live_data: bool | None = None
    use_gpt: bool | None = None
    llm_backend: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ForecasterConfig":
        return cls(
            name=data["name"],
            provider=data["provider"],
            model=data["model"],
            api_key_env=data.get("api_key_env"),
            api_key_fallback_envs=list(data.get("api_key_fallback_envs") or []),
            enabled=bool(data.get("enabled", True)),
            weight=float(data.get("weight", 1.0)),
            temperature=float(data.get("temperature", 0.1)),
            max_tokens=int(data.get("max_tokens", 1400)),
            reasoning_effort=data.get("reasoning_effort"),
            system_prompt=data.get("system_prompt"),
            system_prompt_path=data.get("system_prompt_path"),
            enable_google_search=bool(data.get("enable_google_search", True)),
            require_google_search_grounding=bool(data.get("require_google_search_grounding", False)),
            adapter_config_path=data.get("adapter_config_path"),
            deadline_seconds=None if data.get("deadline_seconds") is None else float(data.get("deadline_seconds")),
            use_live_data=None if data.get("use_live_data") is None else bool(data.get("use_live_data")),
            use_gpt=None if data.get("use_gpt") is None else bool(data.get("use_gpt")),
            llm_backend=data.get("llm_backend"),
        )


def _prep_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_prompt_file(path_value: str) -> str:
    path = Path(path_value)
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, _prep_root() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(f"system_prompt_path not found: {path_value}")


def system_prompt_for_config(config: ForecasterConfig) -> str:
    if config.system_prompt_path:
        return _read_prompt_file(config.system_prompt_path)
    if config.system_prompt:
        return config.system_prompt
    return SYSTEM_PROMPT


def stable_prompt_hash(packet: MarketPacket, config: ForecasterConfig) -> str:
    system_prompt = system_prompt_for_config(config)
    payload = {
        "packet": packet.to_dict(),
        "model": config.model,
        "provider": config.provider,
        "prompt": system_prompt,
        "enable_google_search": config.enable_google_search,
        "require_google_search_grounding": config.require_google_search_grounding,
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


def _market_mirror_model_forecast(
    config: ForecasterConfig,
    packet: MarketPacket,
    reason: str,
) -> ModelForecast:
    """Build a market-mirror ModelForecast for the lane-timeout path.

    Binary YES/NO with Kalshi quote: probabilities track market_mid.
    Multi-outcome: current market-implied probabilities from live Kalshi
    enrichment when available, otherwise uniform across packet.outcomes.

    Marks `should_defer_to_market=True` so the ensemble aggregator weights
    this lane down rather than treating it as an opinionated forecast.
    """
    outs = packet.outcomes or ["YES", "NO"]
    kalshi = getattr(packet, "kalshi", None)
    mid = 0.5
    if kalshi is not None:
        try:
            mid = float(kalshi.market_mid)
        except (TypeError, ValueError, AttributeError):
            mid = 0.5

    if is_yes_no_outcomes(outs):
        probs = {outs[0]: mid, outs[1]: 1.0 - mid}
    else:
        market_probs = packet.retrieval.get("market_implied_probabilities")
        if isinstance(market_probs, dict):
            n = max(1, len(outs))
            probs = {}
            for outcome in outs:
                try:
                    probs[outcome] = float(market_probs.get(outcome, 1.0 / n))
                except (TypeError, ValueError):
                    probs[outcome] = 1.0 / n
            if packet.is_mutually_exclusive:
                probs = normalize_distribution(probs)
        else:
            n = max(1, len(outs))
            probs = {o: 1.0 / n for o in outs}

    response = {
        "forecast": {
            "probabilities": probs,
            "confidence": 0.30,
            "uncertainty": 0.70,
        },
        "reasoning_track": {
            "summary": f"Lane {config.name} hit wall-clock budget ({reason}); mirroring market.",
            "base_rate": "",
            "market_analysis": "Lane timeout: deferring to market price.",
        },
        "diagnostics": {
            "evidence_quality": "low",
            "rules_clarity": "medium",
            "liquidity_quality": "medium",
            "market_disagreement_reason": "",
            "should_defer_to_market": True,
        },
        "lane_timeout": {"reason": reason},
    }
    return forecast_from_response(
        provider=config.provider,
        model_id=config.model,
        packet=packet,
        response=response,
        raw_response={"lane_timeout": {"reason": reason}},
    )


def _dispatch_provider(config: ForecasterConfig):
    """Resolve the provider-specific forecast() callable."""
    if config.provider == "gemini":
        from .gemini import forecast
        return forecast
    if config.provider == "openrouter":
        from .openrouter import forecast
        return forecast
    if config.provider == "grok":
        from .grok import forecast
        return forecast
    if config.provider == "dhruv_gemini":
        from .dhruv_gemini import forecast
        return forecast
    if config.provider == "claude_agent":
        from .claude import forecast
        return forecast
    raise ValueError(f"Unknown forecaster provider: {config.provider}")


def forecast_from_config(config: ForecasterConfig, packet: MarketPacket) -> ModelForecast:
    """Run a provider-specific forecaster under a 7.5-minute wall-clock budget.

    On timeout, returns a market-mirror ModelForecast so the ensemble still
    receives a valid distribution from this lane. Other exceptions (unknown
    provider, missing API key, parse failure) propagate to the caller; the
    agent server's lane fan-out treats those as lane failures and excludes
    them from the aggregate, which already falls back to the market anchor
    when zero lanes succeed.
    """
    provider_forecast = _dispatch_provider(config)

    budget = float(os.environ.get("LANE_TIMEOUT_SECONDS", DEFAULT_LANE_TIMEOUT_SECONDS))
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        fut = pool.submit(provider_forecast, config, packet)
        try:
            return fut.result(timeout=budget)
        except FuturesTimeoutError:
            logger.warning(
                "lane %s (%s) exceeded %.0fs budget; returning market mirror",
                config.name, config.provider, budget,
            )
            return _market_mirror_model_forecast(
                config, packet, f"lane_timeout_{int(budget)}s"
            )
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def clamped_market_plus_edge(packet: MarketPacket, edge_bps: float) -> float:
    return clamp_prob(packet.kalshi.market_mid + edge_bps / 10_000.0)
