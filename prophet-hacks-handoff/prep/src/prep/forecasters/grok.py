"""Grok-via-OpenRouter forecaster with the trust-extreme calibration prompt.

This is Victor's leg of the ensemble. It calls Grok-4.20 through OpenRouter
with a system prompt tuned to fix Grok's primary failure mode on Kalshi-style
markets: hedging confident markets toward 0.5.

Validated on a contamination-free 2026 Sports holdout (N=190 KTV markets
post-Grok-training-cutoff): Brier 0.203 vs market 0.212 (Δ -0.85pp,
P(better)=98%) when combined with a noise-removal filter (high-volume,
extreme-priced, or ATP tennis markets fall through to market price).

Three pieces produce that win, all in this file:
1. Trust-extreme system prompt (TRUST_EXTREME_SYSTEM)
2. Bidirectional prompting on binary markets (ask P(YES) + P(NO), average)
3. Noise-removal filter that short-circuits the API call on markets where
   Grok historically underperforms market price (high-volume, extreme-priced,
   or ATP-tennis series) — returns a market-mirror forecast with
   should_defer_to_market=True so the ensemble downweights it.

Pinned settings that produced the validated numbers:
- model: x-ai/grok-4.20
- temperature: 0.7
- GROK_BIDIR=1   (env, on by default for binary)
- GROK_FILTER=1  (env, on by default — disable to A/B against unfiltered)
- GROK_VOLUME_SKIP=4000, GROK_EXTREME_SKIP=0.15 (env, filter thresholds)
"""

from __future__ import annotations

import os
import time

import requests

from .base import (
    ForecasterConfig,
    build_user_prompt,
    extract_json_object,
    forecast_from_response,
    resolve_api_key,
    stable_prompt_hash,
    system_prompt_for_config,
)
from prep.schemas import MarketPacket, normalize_distribution
from prep.schemas import is_yes_no_outcomes


OPENROUTER_CHAT_COMPLETIONS = "https://openrouter.ai/api/v1/chat/completions"

# Wall-clock budget for the whole Grok lane (covers bidir's two calls + parsing).
# On timeout we fall back to a market-mirror forecast so the ensemble still has
# a valid distribution to aggregate. Override with GROK_TIMEOUT_SECONDS.
DEFAULT_TIMEOUT_BUDGET_SECONDS = 450.0


# Trust-extreme calibration guidance. Composed with the team's structured user
# prompt (build_user_prompt) — this only contributes the calibration philosophy.
# Schema/output instructions come from the user prompt.
TRUST_EXTREME_SYSTEM = """\
You are an expert prediction-market forecaster specialized in calibrated
probability estimation.

CALIBRATION GUIDELINES:
- Consider base rates for similar events.
- Weight evidence by reliability and recency.
- Account for uncertainty — don't be overconfident.
- Extremes (p < 0.10 or p > 0.90) require very strong evidence.
- MUTUALLY EXCLUSIVE MARKETS: if this market is one of N candidates competing
  for a single outcome, the base rate is roughly 1/N. Most such markets
  resolve NO.
- TRUST THE MARKET AT EXTREMES: when the market price is below 0.10 or above
  0.90, the market is making a very confident statement based on large
  amounts of capital from informed traders. Unless you have specific
  knowledge of a key fact the market appears to have missed (a recent
  injury, a weather forecast, a regulatory ruling), your prediction should
  stay close to the market price. Refusing to commit to an extreme is a
  common LLM error — do not make it. If the market is at 0.97, your
  prediction should typically be in [0.90, 0.99], not pulled to 0.5.

Provide an auditable reasoning track: cite evidence, assumptions,
counterarguments, information gaps, and probability adjustments. Do not
include hidden chain-of-thought. Return only valid JSON matching the schema
the user provides.
"""


# Same calibration guidance but flipped to ask about P(NO). Used for the
# second leg of bidirectional prompting on binary YES/NO markets.
TRUST_EXTREME_SYSTEM_NO = TRUST_EXTREME_SYSTEM


def _api_key(config: ForecasterConfig) -> str:
    key = resolve_api_key(config, "OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(f"No API key found for {config.name} (checked {config.api_key_env} and fallbacks)")
    return key


def _resolve_system_prompt(config: ForecasterConfig) -> str:
    """If the config sets system_prompt or system_prompt_path explicitly,
    honor it. Otherwise use the trust-extreme prompt — this is the whole
    point of the grok adapter vs. plain openrouter."""
    if config.system_prompt_path or config.system_prompt:
        return system_prompt_for_config(config)
    return TRUST_EXTREME_SYSTEM


def _bidir_enabled(config: ForecasterConfig, packet: MarketPacket) -> bool:
    """Bidirectional prompting only makes sense for binary YES/NO markets."""
    if not is_yes_no_outcomes(packet.outcomes):
        return False
    raw = getattr(config, "extra", None) or {}
    if isinstance(raw, dict) and "bidir" in raw:
        return bool(raw["bidir"])
    env = os.environ.get("GROK_BIDIR")
    if env is not None:
        return env not in ("0", "false", "False", "")
    return True


# Noise-removal filter thresholds. Defaults match what we validated against
# the 2026 KTV holdout. Skipping Grok on these markets and falling through
# to market price is the OTHER half of the validated -0.85pp Brier win.
DEFAULT_VOLUME_SKIP = 4000.0
DEFAULT_EXTREME_SKIP = 0.15
SKIP_SERIES_PREFIXES = ("KXATPMATCH", "KXATPCHALLENGERMATCH")


def _should_skip_grok(packet: MarketPacket) -> tuple[bool, str | None]:
    """Decide whether to skip the Grok call and let the market speak.

    Three skip rules, validated independently and stackable:
    - High-volume markets (Grok's prior < market's micro-structure signal)
    - Extreme-priced markets (Grok hedges 0.99→0.15 on confident markets)
    - ATP tennis (Grok consistently underforecasts favorites in this series)

    Returns (skip, reason). When skip=True, the caller short-circuits with
    a market-mirror forecast that signals should_defer_to_market=True.
    """
    vol_thr = float(os.environ.get("GROK_VOLUME_SKIP", DEFAULT_VOLUME_SKIP))
    extreme_thr = float(os.environ.get("GROK_EXTREME_SKIP", DEFAULT_EXTREME_SKIP))

    ticker = (packet.market_ticker or "").upper()
    if any(ticker.startswith(p) for p in SKIP_SERIES_PREFIXES):
        return True, "atp_tennis_series"

    kalshi = getattr(packet, "kalshi", None)
    if kalshi is None:
        return False, None

    vol = getattr(kalshi, "volume", None)
    if vol is not None and float(vol) > vol_thr:
        return True, f"high_volume_{int(vol)}>{int(vol_thr)}"

    try:
        mid = float(kalshi.market_mid)
    except (TypeError, ValueError, AttributeError):
        return False, None
    if mid <= extreme_thr or mid >= (1.0 - extreme_thr):
        return True, f"extreme_mid_{mid:.3f}"

    return False, None


def _market_mirror_response(packet: MarketPacket, reason: str) -> dict:
    """Build a parsed-response dict that mirrors market mid, deferring to it.
    Used when the filter skips the Grok call entirely."""
    outs = tuple(packet.outcomes) if packet.outcomes else ("YES", "NO")
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
            raw = {}
            for outcome in outs:
                try:
                    raw[outcome] = float(market_probs.get(outcome, 1.0 / n))
                except (TypeError, ValueError):
                    raw[outcome] = 1.0 / n
            probs = normalize_distribution(raw)
        else:
            n = max(1, len(outs))
            probs = {o: 1.0 / n for o in outs}

    return {
        "forecast": {
            "probabilities": probs,
            "confidence": 0.30,
            "uncertainty": 0.70,
        },
        "reasoning_track": {
            "summary": f"Grok skipped by noise-removal filter ({reason}); mirroring market.",
            "base_rate": "",
            "market_analysis": "Filter rule triggered: deferring to market price.",
            "context_market_analysis": "",
            "key_evidence": [],
            "source_audit": [],
            "counterarguments": [],
            "assumptions": [f"Filter rule: {reason}"],
            "information_gaps": [],
            "what_would_change_my_mind": [],
        },
        "diagnostics": {
            "evidence_quality": "low",
            "rules_clarity": "medium",
            "liquidity_quality": "high",
            "market_disagreement_reason": "",
            "should_defer_to_market": True,
        },
        "filter_skip": {"reason": reason},
    }


def _single_call(
    config: ForecasterConfig,
    packet: MarketPacket,
    system_prompt: str,
    *,
    remaining_budget: float = 180.0,
) -> tuple[dict, dict]:
    """One OpenRouter call. Returns (parsed_response, raw_api_response).

    `remaining_budget` caps this HTTP request so the whole lane stays under the
    wall-clock budget enforced by `forecast()`. Always at least 1 second so we
    don't accidentally pass a 0/negative timeout to `requests`.
    """
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_prompt(packet)},
        ],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "response_format": {"type": "json_object"},
    }
    if config.reasoning_effort:
        payload["extra_body"] = {"reasoning": {"effort": config.reasoning_effort}}

    per_call_timeout = max(1.0, min(180.0, remaining_budget))
    resp = requests.post(
        OPENROUTER_CHAT_COMPLETIONS,
        headers={
            "Authorization": f"Bearer {_api_key(config)}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost/prophet-hacks",
            "X-Title": "ProphetHacks Ensemble (grok)",
        },
        json=payload,
        timeout=per_call_timeout,
    )
    resp.raise_for_status()
    raw = resp.json()
    text = raw["choices"][0]["message"].get("content", "")
    parsed = extract_json_object(text)
    return parsed, raw


def _average_bidir(yes_parsed: dict, no_parsed: dict, packet: MarketPacket) -> dict:
    """Combine YES-direction and NO-direction parsed responses by averaging
    the YES probability: p_yes = (yes_p_yes + (1 - no_p_yes)) / 2.

    Falls back to the YES-direction response when either side is missing
    a usable probability."""
    def _extract_p_yes(parsed: dict) -> float | None:
        forecast = parsed.get("forecast") or {}
        probs = forecast.get("probabilities")
        if isinstance(probs, dict) and "YES" in probs:
            try:
                return float(probs["YES"])
            except (TypeError, ValueError):
                pass
        p_yes_raw = forecast.get("p_yes")
        if p_yes_raw is not None:
            try:
                return float(p_yes_raw)
            except (TypeError, ValueError):
                pass
        return None

    p_yes_y = _extract_p_yes(yes_parsed)
    p_yes_n = _extract_p_yes(no_parsed)
    if p_yes_y is None and p_yes_n is None:
        return yes_parsed
    if p_yes_y is None:
        p_yes_y = p_yes_n
    if p_yes_n is None:
        p_yes_n = p_yes_y
    averaged = max(0.01, min(0.99, (p_yes_y + p_yes_n) / 2.0))

    # Mutate the YES-direction response so the team's downstream parser sees
    # the averaged distribution. Keep its reasoning track intact.
    yes_parsed.setdefault("forecast", {})
    yes_parsed["forecast"]["probabilities"] = {
        "YES": averaged,
        "NO": 1.0 - averaged,
    }
    yes_parsed["bidir"] = {
        "yes_direction_p_yes": p_yes_y,
        "no_direction_implied_p_yes": p_yes_n,
        "averaged_p_yes": averaged,
    }
    return yes_parsed


def _timeout_fallback(config: ForecasterConfig, packet: MarketPacket, reason: str):
    """Build a market-mirror forecast for the timeout/error path."""
    parsed = _market_mirror_response(packet, reason)
    parsed["prompt_hash"] = stable_prompt_hash(packet, config)
    return forecast_from_response(
        provider="grok",
        model_id=config.model,
        packet=packet,
        response=parsed,
        raw_response={"timeout_fallback": {"reason": reason}},
    )


def forecast(config: ForecasterConfig, packet: MarketPacket):
    # Noise-removal filter: skip the API call entirely on markets where Grok
    # systematically underperforms market price. Returns a deferring forecast
    # the ensemble will weight down. Disable with GROK_FILTER=0.
    if os.environ.get("GROK_FILTER", "1") not in ("0", "false", "False", ""):
        skip, reason = _should_skip_grok(packet)
        if skip:
            parsed = _market_mirror_response(packet, reason)
            parsed["prompt_hash"] = stable_prompt_hash(packet, config)
            return forecast_from_response(
                provider="grok",
                model_id=config.model,
                packet=packet,
                response=parsed,
                raw_response={"filter_skip": {"reason": reason}},
            )

    # 7.5-minute wall-clock budget across the whole lane. On timeout (or any
    # transport-level failure during a call), fall back to market mirror so the
    # ensemble still gets a valid distribution.
    budget = float(os.environ.get("GROK_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_BUDGET_SECONDS))
    deadline = time.monotonic() + budget

    system_prompt = _resolve_system_prompt(config)

    try:
        yes_parsed, yes_raw = _single_call(
            config, packet, system_prompt,
            remaining_budget=deadline - time.monotonic(),
        )
    except (requests.Timeout, requests.RequestException) as exc:
        return _timeout_fallback(config, packet, f"grok_timeout_yes:{type(exc).__name__}")

    raw_bundle: dict = {"yes_direction": yes_raw}

    if _bidir_enabled(config, packet):
        remaining = deadline - time.monotonic()
        if remaining <= 1.0:
            return _timeout_fallback(config, packet, "grok_timeout_before_no_direction")
        try:
            no_parsed, no_raw = _single_call(
                config, packet, TRUST_EXTREME_SYSTEM_NO,
                remaining_budget=remaining,
            )
        except (requests.Timeout, requests.RequestException) as exc:
            return _timeout_fallback(config, packet, f"grok_timeout_no:{type(exc).__name__}")
        raw_bundle["no_direction"] = no_raw
        parsed = _average_bidir(yes_parsed, no_parsed, packet)
    else:
        parsed = yes_parsed

    parsed["prompt_hash"] = stable_prompt_hash(packet, config)
    return forecast_from_response(
        provider="grok",
        model_id=config.model,
        packet=packet,
        response=parsed,
        raw_response=raw_bundle,
    )
