"""Zero-shot predictor via OpenRouter, with bi-directional prompting
and (optionally) market context in the prompt.

OpenRouter exposes every frontier model through one OpenAI-compatible API,
which is how Victor's leg of the ensemble runs Grok (and lets us
ad-hoc spelunk any model by changing one env var).

**Bi-directional prompting** (Prophet Arena paper §C.3): instead of one
P(YES) call, we make two — one asking P(YES this resolves YES) and one
asking P(NO this resolves NO) — and average `(p_yes_direct + (1 - p_no_direct)) / 2`.
The paper shows this improves calibration on 4 of 5 frontier models at
the cost of 2x LLM calls.

**Market context in prompt** (paper Fig 5, §4.2.2): when `market_info`
is provided, we render the Kalshi yes_ask / no_ask / implied midpoint
into the prompt. The paper shows this is the single largest measured
intervention (0.235 → 0.173 average Brier, 26% relative improvement),
because the market price already aggregates many traders' information
and frontier LLMs anchor on it well *when given the chance*.

The predictor exposes both signatures (paper-consistent — see
`baselines/market.py`):
    predict(event)                 # no market context (zero-shot prior)
    predict(event, market_info)    # paper's recommended setup

`prep.eval.evaluate()` adapts to whichever signature is needed.

Env:
    OPENROUTER_API_KEY        required
    OPENROUTER_MODEL          default: x-ai/grok-4.20
    OPENROUTER_TEMPERATURE    default: 0.7
    OPENROUTER_MAX_TOKENS     default: 300
    OPENROUTER_BIDIR          set to "0" to disable bi-direction (single P(YES) call)
    OPENROUTER_USE_MARKET     set to "0" to suppress market context even when given
"""

from __future__ import annotations

import json
import os
import sys
import time
from functools import lru_cache

from openai import OpenAI

_MULTI_CANDIDATE_NOTE = """\
- MUTUALLY EXCLUSIVE MARKETS: Many Kalshi markets are one slice of a
  multi-option event (e.g. "Will [candidate X] win [office Y]?" is one of
  N candidate-specific markets for the same election; "Will [team] win
  the championship?" is one of N team-specific markets). Only one option
  resolves YES — so the base rate for any specific option is roughly 1/N
  unless you have strong evidence that option is the favorite. Asking
  yourself "how many other candidates / teams / outcomes is this
  competing against?" is the single most important calibration check
  for these markets. Most such markets resolve NO."""

SYSTEM_YES = f"""\
You are an expert forecaster specialized in calibrated probability estimation.

Your task is to estimate the probability that the given event resolves YES.

CALIBRATION GUIDELINES:
- Consider base rates for similar events.
- Weight evidence by reliability and recency.
- Account for uncertainty — don't be overconfident.
- Extremes (p < 0.10 or p > 0.90) require very strong evidence.
{_MULTI_CANDIDATE_NOTE}

Respond with ONLY valid JSON: {{"p_yes": <float 0.01-0.99>, "rationale": "<2-3 sentences>"}}
Do not include any other text outside the JSON object."""

SYSTEM_NO = f"""\
You are an expert forecaster specialized in calibrated probability estimation.

Your task is to estimate the probability that the given event resolves NO.

CALIBRATION GUIDELINES:
- Consider base rates for similar events.
- Weight evidence by reliability and recency.
- Account for uncertainty — don't be overconfident.
- Extremes (p < 0.10 or p > 0.90) require very strong evidence.
{_MULTI_CANDIDATE_NOTE}

Respond with ONLY valid JSON: {{"p_no": <float 0.01-0.99>, "rationale": "<2-3 sentences>"}}
Do not include any other text outside the JSON object."""


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    return OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")


def _format_market_context(market_info: dict | None) -> str | None:
    """Render market_info into a prompt block. Returns None if unusable.

    Kalshi prices are in cents (0-100). yes_ask = cost to buy YES;
    no_ask = cost to buy NO. (yes_ask + (100 - no_ask)) / 200 is the
    spread-corrected market-implied P(YES).
    """
    if not market_info:
        return None
    yes_ask = market_info.get("yes_ask")
    no_ask = market_info.get("no_ask")
    last_price = market_info.get("last_price")
    # Liquidity proxy: prefer 24h volume if available; fall back to all-time
    # volume; finally to Kalshi `liquidity` (depth-weighted notional, only
    # field carried by the Subset-1200 dataset).
    liquidity_signal = (
        market_info.get("volume_24h")
        or market_info.get("volume")
        or market_info.get("liquidity")
    )

    p_implied: float | None = None
    if yes_ask is not None and no_ask is not None and (yes_ask + no_ask) > 0:
        p_implied = (yes_ask + (100 - no_ask)) / 200
    elif last_price is not None:
        p_implied = last_price / 100

    if p_implied is None:
        return None

    lines = ["", "CURRENT KALSHI MARKET PRICE (use as a strong prior — see guidance):"]
    if yes_ask is not None and no_ask is not None:
        p_yes_lower = yes_ask / 100
        p_yes_upper = (100 - no_ask) / 100
        lines.append(f"- Top-of-book YES ask: {yes_ask}¢  (implies market thinks P(YES) ≥ {p_yes_lower:.3f})")
        lines.append(f"- Top-of-book NO ask:  {no_ask}¢   (implies market thinks P(YES) ≤ {p_yes_upper:.3f})")
    if last_price is not None:
        lines.append(f"- Last trade: {last_price}¢")
    lines.append(f"- Market-implied P(YES) (spread-corrected midpoint): {p_implied:.3f}")
    if liquidity_signal is not None:
        lines.append(f"- Liquidity proxy: {liquidity_signal:g}")
    return "\n".join(lines)


def _build_user_prompt(event: dict, market_info: dict | None = None) -> str:
    parts = [f"Event: {event['title']}"]
    if event.get("subtitle"):
        parts.append(f"Subtitle: {event['subtitle']}")
    if event.get("description"):
        parts.append(f"Description: {event['description']}")
    if event.get("rules"):
        parts.append(f"Rules: {event['rules']}")
    parts.append(f"Category: {event['category']}")
    parts.append(f"Close time: {event['close_time']}")

    use_market = os.environ.get("OPENROUTER_USE_MARKET", "1") != "0"
    market_block = _format_market_context(market_info) if use_market else None
    if market_block:
        parts.append(market_block)
        parts.append(
            "\nGUIDANCE: The market price reflects the collective view of many "
            "traders with skin in the game and is a strong prior. You should "
            "anchor on it but you are NOT required to copy it. Deviate from "
            "the market only when you can articulate specific information or "
            "reasoning the market appears to have missed; otherwise stay close "
            "to the market-implied probability."
        )

    return "\n".join(parts)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _ask(system: str, user: str, key: str, retries: int = 2) -> tuple[float, str] | None:
    """Single LLM call. Returns (p, rationale) or None on unrecoverable failure.

    Retries on network errors and parse failures with light backoff. A
    parse failure usually means the model returned empty content or
    prose around the JSON — both transient enough that a re-roll fixes
    them most of the time.
    """
    model = os.environ.get("OPENROUTER_MODEL", "x-ai/grok-4.20")
    temperature = float(os.environ.get("OPENROUTER_TEMPERATURE", "0.7"))
    max_tokens = int(os.environ.get("OPENROUTER_MAX_TOKENS", "300"))

    last_err: str | None = None
    for attempt in range(retries + 1):
        try:
            resp = _client().chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            text = _strip_fences(resp.choices[0].message.content or "")
            if not text:
                last_err = "empty content"
                time.sleep(0.5 * (attempt + 1))
                continue
            data = json.loads(text)
            p = float(data[key])
            return max(0.01, min(0.99, p)), data.get("rationale", "")
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(0.5 * (attempt + 1))
        except Exception as e:  # noqa: BLE001 — keep going on transient API errors
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(1.0 * (attempt + 1))
    sys.stderr.write(f"[openrouter] giving up after {retries + 1} attempts: {last_err}\n")
    return None


def predict(event: dict, market_info: dict | None = None) -> dict:
    user_prompt = _build_user_prompt(event, market_info)
    bidir = os.environ.get("OPENROUTER_BIDIR", "1") != "0"

    if not bidir:
        result = _ask(SYSTEM_YES, user_prompt, "p_yes")
        if result is None:
            return {"p_yes": 0.5, "rationale": "fallback: predictor failed"}
        p_yes, rationale = result
        return {"p_yes": p_yes, "rationale": rationale}

    yes_result = _ask(SYSTEM_YES, user_prompt, "p_yes")
    no_result = _ask(SYSTEM_NO, user_prompt, "p_no")

    # Graceful degradation: if one side fails, use the other; if both
    # fail, return 0.5 (small Brier hit but keeps the run alive).
    if yes_result is not None and no_result is not None:
        p_yes_direct, why_yes = yes_result
        p_no_direct, why_no = no_result
        p_yes = (p_yes_direct + (1 - p_no_direct)) / 2
        rationale = (
            f"bi-dir avg of P(YES)={p_yes_direct:.3f}, 1-P(NO)={1 - p_no_direct:.3f}. "
            f"YES-rationale: {why_yes} NO-rationale: {why_no}"
        )
    elif yes_result is not None:
        p_yes, _ = yes_result
        rationale = "fallback: NO-direction predictor failed, using P(YES) only"
    elif no_result is not None:
        p_no, _ = no_result
        p_yes = 1 - p_no
        rationale = "fallback: YES-direction predictor failed, using 1-P(NO) only"
    else:
        p_yes = 0.5
        rationale = "fallback: both predictors failed"

    return {"p_yes": p_yes, "rationale": rationale}
