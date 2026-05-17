"""Grok variant: prompt explicitly tells the model to trust market price
at extremes (≥0.95 or ≤0.05).

Diagnosis from 2026 x10 sample: Grok's biggest Brier losses are markets
where market price is at an extreme (very confident) and Grok hedges toward
0.5. E.g., market 0.99, Grok 0.15, truth YES → Brier loss 0.84.

This prompt aims to make Grok defer to extreme market prices unless it has
very specific contradicting evidence.
"""

from __future__ import annotations

import json
import os
import sys
import time
from functools import lru_cache

from openai import OpenAI

SYSTEM_YES = """\
You are an expert forecaster specialized in calibrated probability estimation.

Your task is to estimate the probability that the given event resolves YES.

CALIBRATION GUIDELINES:
- Consider base rates for similar events.
- Weight evidence by reliability and recency.
- Account for uncertainty — don't be overconfident.
- Extremes (p < 0.10 or p > 0.90) require very strong evidence.
- MUTUALLY EXCLUSIVE MARKETS: If this is one of N candidates competing,
  the base rate is roughly 1/N. Most such markets resolve NO.
- TRUST THE MARKET AT EXTREMES: When the market price is below 0.10 or
  above 0.90, the market is making a very confident statement based on
  large amounts of capital from informed traders. Unless you have specific
  knowledge of a key fact the market appears to have missed (e.g. a recent
  injury, a weather forecast, a regulatory ruling), your prediction should
  stay close to the market price. Refusing to commit to an extreme is a
  common LLM error — do not make it. If market is at 0.97, your prediction
  should typically be in [0.90, 0.99], not pulled to 0.5.

Respond with ONLY valid JSON: {"p_yes": <float 0.01-0.99>, "rationale": "<2-3 sentences>"}
Do not include any other text outside the JSON object."""

SYSTEM_NO = SYSTEM_YES.replace("resolves YES", "resolves NO").replace('"p_yes"', '"p_no"')


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    return OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")


def _format_market_context(market_info: dict | None) -> str | None:
    if not market_info:
        return None
    yes_ask = market_info.get("yes_ask"); no_ask = market_info.get("no_ask")
    last_price = market_info.get("last_price")
    p = None
    if yes_ask is not None and no_ask is not None and (yes_ask + no_ask) > 0:
        p = (yes_ask + (100 - no_ask)) / 200
    elif last_price is not None:
        p = last_price / 100
    if p is None:
        return None
    lines = ["", "CURRENT KALSHI MARKET PRICE (use as a STRONG prior — see guidance):"]
    if yes_ask is not None and no_ask is not None:
        lines.append(f"- YES ask: {yes_ask}¢   NO ask: {no_ask}¢")
    lines.append(f"- Market-implied P(YES) (midpoint): {p:.3f}")
    if p >= 0.90 or p <= 0.10:
        lines.append(f"- ⚠ MARKET IS AT AN EXTREME ({p:.2f}). Strongly consider matching it unless you have specific evidence it has missed something.")
    return "\n".join(lines)


def _build_user_prompt(event: dict, market_info: dict | None) -> str:
    parts = [f"Event: {event['title']}"]
    if event.get("subtitle"): parts.append(f"Subtitle: {event['subtitle']}")
    if event.get("description"): parts.append(f"Description: {event['description']}")
    if event.get("rules"): parts.append(f"Rules: {event['rules']}")
    parts.append(f"Category: {event['category']}")
    parts.append(f"Close time: {event['close_time']}")
    mb = _format_market_context(market_info)
    if mb:
        parts.append(mb)
    return "\n".join(parts)


def _ask(system, user, key, retries=2):
    model = os.environ.get("OPENROUTER_MODEL", "x-ai/grok-4.20")
    temperature = float(os.environ.get("OPENROUTER_TEMPERATURE", "0.7"))
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = _client().chat.completions.create(
                model=model, max_tokens=300, temperature=temperature,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
            )
            text = (r.choices[0].message.content or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
            data = json.loads(text)
            p = max(0.01, min(0.99, float(data[key])))
            return p, data.get("rationale", "")
        except Exception as e:
            last_err = e
            time.sleep(0.5 * (attempt + 1))
    sys.stderr.write(f"[trust_extreme] giving up: {last_err}\n")
    return None


def predict(event: dict, market_info: dict | None = None) -> dict:
    user = _build_user_prompt(event, market_info)
    bidir = os.environ.get("OPENROUTER_BIDIR", "1") != "0"
    if not bidir:
        r = _ask(SYSTEM_YES, user, "p_yes")
        if r is None: return {"p_yes": 0.5, "rationale": "fallback"}
        return {"p_yes": r[0], "rationale": r[1]}
    y = _ask(SYSTEM_YES, user, "p_yes")
    n = _ask(SYSTEM_NO, user, "p_no")
    if y and n:
        p = (y[0] + (1 - n[0])) / 2
        return {"p_yes": p, "rationale": f"bidir {y[0]:.3f}/{n[0]:.3f}"}
    if y: return {"p_yes": y[0], "rationale": y[1]}
    if n: return {"p_yes": 1 - n[0], "rationale": n[1]}
    return {"p_yes": 0.5, "rationale": "both failed"}
