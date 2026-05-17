"""Self-deferring Grok predictor.

Strategy distinct from what a parallel agent is likely doing (which is testing
zero-shot reasoning model variants on heterogeneous samples). This one keeps
Grok-4.3 but flips the prompting incentive: instead of asking Grok to produce
its best forecast, ask it to **output the market price exactly unless it has
specific evidence the market has missed**. Then Grok's deviation is conditioned
on Grok's own confidence that it has an edge.

Rationale from our prior audit: zero-shot Grok's deviations from market are
essentially noise (residual correlation with outcome ≈ 0.0007). But maybe
SOME of those deviations are real signal that gets washed out by noisy
deviations on other markets. If we let Grok say "I have no edge here" and
output market price, we filter out noise and keep only the genuine signal.

Env: same as openrouter_zero_shot; bidir is disabled (the deferring instruction
is asymmetric YES/NO so bi-directional doesn't make sense).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from functools import lru_cache

from openai import OpenAI

SYSTEM = """\
You are a forecaster evaluating a binary prediction market on Kalshi. The market
price has been set by many traders with skin in the game and reflects their
aggregated view of the probability.

Your task: output a probability that the event resolves YES. The DEFAULT answer
is to copy the market price exactly. You should only DEVIATE from the market
price when you can articulate SPECIFIC, CONCRETE information that the market
appears to have missed or mispriced. Vague reasoning, base-rate guesses, or
"the market seems wrong" without evidence is NOT sufficient justification.

CALIBRATION RULES:
- If you have no specific information beyond what's in the market, output the
  market_mid exactly.
- If you have moderate evidence the market is mispriced, deviate by 5-15
  percentage points and explain the specific evidence.
- If you have strong, named evidence (e.g. a specific verified news event the
  market hasn't priced), deviate by up to 30 percentage points.
- Extreme deviations (>30 pp) require multiple converging concrete pieces of
  evidence.

Respond with ONLY valid JSON: {"p_yes": <float 0.01-0.99>, "deviated": <bool>, "rationale": "<2-3 sentences>"}
If deviated=false, the rationale must say "no specific edge" or equivalent.
"""


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    return OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")


def _market_mid(market_info: dict | None) -> float | None:
    if not market_info:
        return None
    yes_ask = market_info.get("yes_ask")
    no_ask = market_info.get("no_ask")
    last_price = market_info.get("last_price")
    if yes_ask is not None and no_ask is not None and (yes_ask + no_ask) > 0:
        return (yes_ask + (100 - no_ask)) / 200
    if last_price is not None:
        return last_price / 100
    return None


def _build_prompt(event: dict, market_info: dict | None) -> str:
    mid = _market_mid(market_info)
    parts = [f"Event: {event.get('title','')}"]
    if event.get("subtitle"):
        parts.append(f"Subtitle: {event['subtitle']}")
    if event.get("rules"):
        parts.append(f"Rules: {event['rules']}")
    parts.append(f"Category: {event.get('category','')}")
    parts.append(f"Close time: {event.get('close_time','')}")

    if market_info and mid is not None:
        yes_ask = market_info.get("yes_ask"); no_ask = market_info.get("no_ask")
        parts.append("")
        parts.append("MARKET STATE:")
        if yes_ask is not None and no_ask is not None:
            parts.append(f"  YES ask: {yes_ask}¢   NO ask: {no_ask}¢")
        parts.append(f"  market_mid (your default answer): {mid:.3f}")
        if market_info.get("volume"):
            parts.append(f"  volume: {market_info['volume']}")

    parts.append("")
    parts.append("Remember: output the market_mid exactly unless you have specific evidence to deviate.")
    return "\n".join(parts)


def predict(event: dict, market_info: dict | None = None) -> dict:
    mid = _market_mid(market_info)
    if mid is None:
        return {"p_yes": 0.5, "rationale": "no market price; fallback 0.5"}

    model = os.environ.get("OPENROUTER_MODEL", "x-ai/grok-4.3")
    temperature = float(os.environ.get("OPENROUTER_TEMPERATURE", "0.3"))  # lower for deterministic deferral

    user_prompt = _build_prompt(event, market_info)

    last_err = None
    for attempt in range(3):
        try:
            resp = _client().chat.completions.create(
                model=model, max_tokens=300, temperature=temperature,
                response_format={"type": "json_object"},
                messages=[{"role":"system","content":SYSTEM},
                          {"role":"user","content":user_prompt}],
            )
            text = (resp.choices[0].message.content or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
            data = json.loads(text)
            p_raw = max(0.01, min(0.99, float(data["p_yes"])))
            return {
                "p_yes": p_raw,
                "deviated": bool(data.get("deviated", False)),
                "rationale": data.get("rationale", ""),
                "market_mid": mid,
            }
        except Exception as e:
            last_err = e
            time.sleep(0.5 * (attempt + 1))
    sys.stderr.write(f"[deferring] failed after 3 attempts: {last_err}\n")
    return {"p_yes": mid, "rationale": f"fallback to market: {last_err}", "market_mid": mid, "deviated": False}
