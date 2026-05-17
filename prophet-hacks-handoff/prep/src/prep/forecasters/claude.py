"""Claude forecaster — renders agent prompt and calls OpenRouter for a single completion.

Provider names : "claude_agent", 
Required env   : OPENROUTER_API_KEY  (or whatever api_key_env points to)

Config extras (beyond base ForecasterConfig):
  backtest_mode         : bool  – inject evidence-cutoff block; date-restrict searches
  evidence_cutoff       : str   – ISO-8601 UTC; "auto" uses packet.as_of
  agent_prompt          : str   – markdown filename under forecasters/agents/
  use_polymarket_prior  : bool  – include Polymarket in prior (map.csv only)
  polymarket_map_only   : bool  – never runtime Gamma match during forecast
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import requests

from ..calibration import time_to_close_hours
from ..polymarket import get_market_priors
from ..schemas import (
    MarketPacket,
    ModelForecast,
    clamp_prob,
)

# Maximum Kalshi–Polymarket mid gap before excluding the Poly prior (8pp)
_MAX_KALSHI_POLY_GAP = 0.08
from .base import (
    ForecasterConfig,
    forecast_from_response,
    resolve_api_key,
    stable_prompt_hash,
)
from .openrouter import OPENROUTER_CHAT_COMPLETIONS

AGENTS_DIR = Path(__file__).parent / "prompts"


class _CostTracker:
    def __init__(self) -> None:
        self._total: float = 0.0
        self._last: float | None = None

    def add(self, cost: float) -> None:
        self._total += cost
        self._last = cost

    @property
    def last(self) -> float | None:
        return self._last

    @property
    def total(self) -> float:
        return self._total


_cost_tracker = _CostTracker()


_PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "claude_agent": {
        "agent_prompt": "claude_agent.md",
        "use_polymarket_prior": True,
    }
}

_USER_PROMPT = (
    "Analyze this market following the instructions above. "
    "Output your forecast as a single JSON object exactly matching the output schema. "
    "No other text before or after the JSON."
)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

def _resolve_agent_prompt(config: ForecasterConfig) -> str:
    if config.agent_prompt:
        return config.agent_prompt
    return str(_PROVIDER_DEFAULTS.get(config.provider, _PROVIDER_DEFAULTS["claude_agent"])["agent_prompt"])


def _load_template(config: ForecasterConfig) -> str:
    path = AGENTS_DIR / _resolve_agent_prompt(config)
    raw = path.read_text()
    if raw.startswith("<!--"):
        end = raw.find("-->")
        if end >= 0:
            raw = raw[end + 3:].lstrip()
    return raw


def _render(template: str, mode_block: str, market_json: str) -> str:
    return template.replace("{mode_block}", mode_block).replace("{market_json}", market_json)


# ---------------------------------------------------------------------------
# Mode blocks
# ---------------------------------------------------------------------------

_LIVE_MODE = """\
## Mode: LIVE
Use information through the current date. Recency matters — fresh data the
market may not have fully absorbed is where research-driven edge comes from."""

_BACKTEST_MODE = """\
## Mode: BACKTEST — evidence cutoff {cutoff}
You are replaying this market **as of {cutoff}**. You must internally constrain
your reasoning:
- Do not reference world events after {cutoff}, even if known from training.
- Do not anchor on what "actually happened" — you do not know it.
- When uncertain whether a fact is pre- or post-cutoff, omit it.
Treat the cutoff as a hard wall."""


# ---------------------------------------------------------------------------
# Prior assembly
# ---------------------------------------------------------------------------

_TTC_WEIGHT = {"far": 0.70, "near": 0.85, "close": 0.95, "imminent": 0.99}


def _compute_prior_weight(spread: float | None, depth: float, ttc_band: str) -> float:
    spread_score = math.exp(-spread / 0.08) if spread is not None else 0.07
    depth_bonus = min(0.15, depth / 5000.0)
    base = min(0.95, spread_score + depth_bonus)
    return max(0.05, min(0.97, base * _TTC_WEIGHT.get(ttc_band, 0.70)))


def _microprice(bid: float, ask: float, bid_sz: float, ask_sz: float) -> float:
    if bid_sz + ask_sz < 1.0:
        return (bid + ask) / 2.0
    spread = ask - bid
    raw = (bid_sz * ask + ask_sz * bid) / (bid_sz + ask_sz)
    mid = (bid + ask) / 2.0
    return mid + (raw - mid) * math.exp(-spread / 0.05)


def _compute_prior(
    packet: MarketPacket,
    backtest_mode: bool,
    use_polymarket: bool,
) -> tuple[float, float, str]:
    """Returns (p_prior, sigma, poly_note) blending Kalshi + Polymarket when available."""
    q = packet.kalshi
    bid = q.yes_bid or 0.0
    ask = q.yes_ask or q.market_mid
    bid_sz = getattr(q, "yes_bid_size", None) or 0.0
    ask_sz = getattr(q, "yes_ask_size", None) or 0.0
    spread = q.spread or 0.0

    k_micro = clamp_prob(q.multilevel_microprice(n=3) if q.yes_bid_levels else (_microprice(bid, ask, bid_sz, ask_sz) if (bid and ask) else q.market_mid))
    depth = bid_sz + ask_sz
    k_weight = math.exp(-spread / 0.05) * (min(1.0, depth / 5000) if depth > 0 else 0.5)
    k_weight = max(0.1, k_weight)

    poly_note = "skipped in backtest mode" if backtest_mode else "no Polymarket match (not in map.csv)"
    poly_mid, poly_weight = 0.0, 0.0

    if use_polymarket and not backtest_mode:
        try:
            priors = get_market_priors(packet)
            if priors:
                pq = priors[0].quote
                pm_mid = pq.market_mid
                pm_spread = pq.spread or 0.20
                alignment = abs(pm_mid - k_micro)
                if alignment > _MAX_KALSHI_POLY_GAP:
                    poly_note = (
                        f"excluded: Kalshi–Poly gap {alignment * 100:.1f}pp "
                        f"> {_MAX_KALSHI_POLY_GAP * 100:.0f}pp"
                    )
                else:
                    pw = math.exp(-pm_spread / 0.05) * math.exp(-(alignment ** 2) / 0.02)
                    if pw > 0.05:
                        poly_mid, poly_weight = pm_mid, pw
                        poly_note = (
                            f"mid {pm_mid:.3f}, spread {pm_spread * 100:.1f}pp, "
                            f"weight {pw:.2f} (map.csv)"
                        )
                    else:
                        poly_note = f"low alignment/liquidity (weight {pw:.3f}); excluded"
        except Exception as exc:
            poly_note = f"lookup failed: {exc}"

    total_w = k_weight + poly_weight
    p_prior = clamp_prob((k_weight * k_micro + poly_weight * poly_mid) / total_w)
    sigma = min(max(spread / 2.0, 0.02) + abs(poly_mid - k_micro) * 0.4, 0.45)
    return p_prior, sigma, poly_note


# ---------------------------------------------------------------------------
# Regime classifier
# ---------------------------------------------------------------------------

_LIQ_EXPLAIN = {
    "liquid":    "Deep, tight book. Prior is high-confidence. Deviate only with strong evidence.",
    "mid":       "Moderate depth and spread. Prior is moderately reliable.",
    "illiquid":  "Thin book. Prior price may be far from fair value. Research carries more weight.",
    "no_market": "No meaningful order book. Prior is a placeholder — estimate from first principles.",
}

_TTC_EXPLAIN = {
    "far":      "More than 72 hours to close. Full research depth appropriate.",
    "near":     "24–72 hours. Research can add edge for news-sensitive questions.",
    "close":    "Under 24 hours. Only strong, specific, recent evidence justifies deviation.",
    "imminent": "Under 3 hours. Markets outperform research in this window. Lean on prior.",
}


def _classify(packet: MarketPacket) -> dict[str, Any]:
    q = packet.kalshi
    spread = q.spread
    oi = q.open_interest or 0
    depth = (getattr(q, "yes_bid_size", 0) or 0) + (getattr(q, "yes_ask_size", 0) or 0)

    if spread is None:
        liq = "no_market"
    elif spread <= 0.04 and (depth > 200 or oi > 500):
        liq = "liquid"
    elif spread <= 0.10 and (depth > 50 or oi > 100):
        liq = "mid"
    elif spread <= 0.20:
        liq = "illiquid"
    else:
        liq = "no_market"

    hours = time_to_close_hours(packet)
    if hours is None or hours > 72:
        ttc = "far"
    elif hours > 24:
        ttc = "near"
    elif hours > 3:
        ttc = "close"
    else:
        ttc = "imminent"

    recency_block = ""
    if hours is not None and hours < 6 and liq in ("liquid", "mid"):
        recency_block = (
            "## News-driven carve-out\n"
            "If you detect ALL THREE: price moved >10pp in last hour, TTC <6h, "
            "spread widened — run one recency search for news in the last 6 hours."
        )

    return {
        "liquidity": liq,
        "ttc_band": ttc,
        "ttc_hours": hours or 999.0,
        "regime_explanation": _LIQ_EXPLAIN[liq],
        "ttc_explanation": _TTC_EXPLAIN[ttc],
        "recency_carveout_block": recency_block,
        "depth_total": depth,
        "spread_pp": (spread or 0.0) * 100,
    }


# ---------------------------------------------------------------------------
# OpenRouter call
# ---------------------------------------------------------------------------

def _openrouter_api_key(config: ForecasterConfig) -> str:
    key = resolve_api_key(config, "OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(f"No API key found for {config.name} (checked {config.api_key_env} and fallbacks)")
    return key


def _call_openrouter(system: str, config: ForecasterConfig) -> str:
    payload: dict[str, Any] = {
        "model": config.model,
        "max_tokens": config.max_tokens or 4000,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": _USER_PROMPT},
        ],
    }
    if config.reasoning_effort:
        payload["reasoning"] = {"effort": config.reasoning_effort}

    resp = requests.post(
        OPENROUTER_CHAT_COMPLETIONS,
        headers={
            "Authorization": f"Bearer {_openrouter_api_key(config)}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=480,
    )
    resp.raise_for_status()
    body = resp.json()
    cost = resp.headers.get("x-openrouter-cost") or (body.get("usage") or {}).get("cost")
    if cost is not None:
        _cost_tracker.add(float(cost))
    return body["choices"][0]["message"]["content"]


def _extract_json(text: str) -> dict[str, Any]:
    """Find the last outermost JSON object in text by scanning backward from the final '}'."""
    text = text.strip()
    end = text.rfind("}")
    if end < 0:
        raise ValueError("No JSON object found in response")
    # Walk backward counting brace depth to find the matching opening brace
    depth = 0
    start = -1
    for i in range(end, -1, -1):
        c = text[i]
        if c == "}":
            depth += 1
        elif c == "{":
            depth -= 1
            if depth == 0:
                start = i
                break
    if start < 0:
        raise ValueError("No balanced JSON object found in response")
    return json.loads(text[start:end + 1])


# ---------------------------------------------------------------------------
# ModelForecast builders
# ---------------------------------------------------------------------------

def _prior_result(config: ForecasterConfig, packet: MarketPacket, p_prior: float, reason: str) -> ModelForecast:
    fallback = {
        "forecast": {"p_yes": p_prior, "confidence": 0.3, "uncertainty": 0.5},
        "reasoning_track": {"summary": f"Deferred to market prior. {reason}", "market_analysis": "No deviation from prior."},
        "diagnostics": {"evidence_quality": "low", "should_defer_to_market": True},
    }
    return forecast_from_response(
        provider=config.provider,
        model_id=config.model,
        packet=packet,
        response=fallback,
    )


def _parse_response(config: ForecasterConfig, packet: MarketPacket, raw: dict, p_prior: float) -> ModelForecast:
    raw["prompt_hash"] = stable_prompt_hash(packet, config)
    # Inject market_mid as p_yes fallback so forecast_from_response has something to work with
    fc = raw.setdefault("forecast", {})
    if "p_yes" not in fc and "probabilities" not in fc:
        fc["p_yes"] = p_prior
    return forecast_from_response(
        provider=config.provider,
        model_id=config.model,
        packet=packet,
        response=raw,
        raw_response=raw,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def forecast(config: ForecasterConfig, packet: MarketPacket) -> ModelForecast:
    backtest_mode: bool = getattr(config, "backtest_mode", False)
    evidence_cutoff: str | None = getattr(config, "evidence_cutoff", None)
    if evidence_cutoff == "auto":
        evidence_cutoff = packet.as_of

    use_polymarket = bool(getattr(config, "use_polymarket_prior", True))
    p_prior, _, poly_note = _compute_prior(packet, backtest_mode, use_polymarket)
    ttc_hours = time_to_close_hours(packet)

    q = packet.kalshi
    spread_pp = round((q.spread or 0.0) * 100, 1)
    depth_usd = (getattr(q, "yes_bid_size", None) or 0.0) + (getattr(q, "yes_ask_size", None) or 0.0)

    mode_block = (
        _BACKTEST_MODE.format(cutoff=evidence_cutoff)
        if backtest_mode and evidence_cutoff
        else _LIVE_MODE
    )

    market_json = json.dumps({
        "title": packet.title or "",
        "subtitle": packet.subtitle or "",
        "category": packet.category or "Other",
        "outcomes": packet.outcomes,
        "rules": packet.rules or "",
        "description": packet.retrieval.get("description") or "",
        "close_time": packet.close_time or "",
        "market": {
            "yes_bid": q.yes_bid,
            "yes_ask": q.yes_ask,
            "mid": round(p_prior, 3),
            "spread_pp": spread_pp,
            "open_interest": q.open_interest,
            "depth_usd": round(depth_usd, 0),
            "ttc_hours": round(ttc_hours, 1) if ttc_hours is not None else None,
            "polymarket": poly_note,
        },
    }, indent=2)

    system = _render(_load_template(config), mode_block=mode_block, market_json=market_json)

    try:
        raw_text = _call_openrouter(system, config)
        raw = _extract_json(raw_text)
        result = _parse_response(config, packet, raw, p_prior)
    except requests.Timeout:
        result = _prior_result(config, packet, p_prior, "Timeout after 8 minutes — returning microprice prior")
    except Exception as exc:
        result = _prior_result(config, packet, p_prior, f"API error: {exc}")

    if _cost_tracker.last is not None:
        print(f"    cost: ${_cost_tracker.last:.4f}  total: ${_cost_tracker.total:.4f}", flush=True)

    return result
