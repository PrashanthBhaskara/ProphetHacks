"""Grok-via-OpenRouter forecaster with the trust-extreme calibration prompt.

This is Victor's leg of the ensemble. It calls Grok-4.20 through OpenRouter
with a system prompt tuned to fix Grok's primary failure mode on Kalshi-style
markets: hedging confident markets toward 0.5.

Validated on a contamination-free 2026 Sports holdout (N=190 KTV markets
post-Grok-training-cutoff): Brier 0.203 vs market 0.212 (Δ -0.85pp,
P(better)=98%) when combined with a noise-removal filter (high-volume,
extreme-priced, or ATP tennis markets fall through to market price).

The trust-extreme system prompt is the actual lift. The filter is an
ensemble-layer concern (which markets get LLM calls vs fall through to
market mid) and belongs upstream of this file — see README for the full
pipeline integration.

Pinned settings that produced the validated numbers:
- model: x-ai/grok-4.20
- temperature: 0.7
- bidir: ask P(YES) and separately P(NO), then average. On by default for
  binary YES/NO markets, off for multi-outcome.

Set bidir=false in the ForecasterConfig (via a `bidir` key in the from_dict
JSON) to disable bidirectional prompting and save the second API call.
"""

from __future__ import annotations

import os

import requests

from .base import (
    ForecasterConfig,
    build_user_prompt,
    extract_json_object,
    forecast_from_response,
    stable_prompt_hash,
    system_prompt_for_config,
)
from prep.schemas import MarketPacket


OPENROUTER_CHAT_COMPLETIONS = "https://openrouter.ai/api/v1/chat/completions"


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
    env_name = config.api_key_env or "OPENROUTER_API_KEY"
    key = os.environ.get(env_name)
    if not key:
        raise RuntimeError(f"{env_name} is not set")
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
    if tuple(packet.outcomes) != ("YES", "NO"):
        return False
    raw = getattr(config, "extra", None) or {}
    if isinstance(raw, dict) and "bidir" in raw:
        return bool(raw["bidir"])
    env = os.environ.get("GROK_BIDIR")
    if env is not None:
        return env not in ("0", "false", "False", "")
    return True


def _single_call(
    config: ForecasterConfig,
    packet: MarketPacket,
    system_prompt: str,
) -> tuple[dict, dict]:
    """One OpenRouter call. Returns (parsed_response, raw_api_response)."""
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

    resp = requests.post(
        OPENROUTER_CHAT_COMPLETIONS,
        headers={
            "Authorization": f"Bearer {_api_key(config)}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost/prophet-hacks",
            "X-Title": "ProphetHacks Ensemble (grok)",
        },
        json=payload,
        timeout=180,
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


def forecast(config: ForecasterConfig, packet: MarketPacket):
    system_prompt = _resolve_system_prompt(config)

    yes_parsed, yes_raw = _single_call(config, packet, system_prompt)

    raw_bundle: dict = {"yes_direction": yes_raw}

    if _bidir_enabled(config, packet):
        no_parsed, no_raw = _single_call(config, packet, TRUST_EXTREME_SYSTEM_NO)
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
