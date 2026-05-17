"""Event-aware predictor: prompts the LLM with ALL sibling markets at once.

Addresses the failure mode found in `openrouter_zero_shot` on Politics
markets: when asked individually, "Will [candidate X] win [office Y]?"
gets answered ~85% YES for every candidate, because the LLM treats each
market in isolation.

This module groups markets by `event_ticker` and asks the LLM to
predict a probability for EACH outcome at once, with an explicit
sum-to-1 constraint. The output for the requested market is then
returned (and normalized).

Two entrypoints:

    predict_event(event_ticker, sibling_markets, market_info_by_ticker)
        → {ticker: p_yes} for the whole event

    predict(event, market_info, candidate_set=None)
        → {p_yes, rationale} for a single market (production-style)
          When `candidate_set` is provided, siblings are extracted from
          it. Otherwise this falls back to single-market prompting (same
          as openrouter_zero_shot).

Env (same as openrouter_zero_shot):
    OPENROUTER_API_KEY        required
    OPENROUTER_MODEL          default: x-ai/grok-4.20
    OPENROUTER_TEMPERATURE    default: 0.5 (lower than zero-shot for stability across siblings)
    OPENROUTER_MAX_TOKENS     default: 600 (more tokens for multi-output JSON)
"""

from __future__ import annotations

import json
import os
import sys
import time
from functools import lru_cache

from openai import OpenAI

SYSTEM = """\
You are an expert forecaster. You will be shown a multi-outcome event from
Kalshi (a real-money prediction market) with all its constituent binary
markets. Estimate the probability that EACH outcome resolves YES.

CALIBRATION GUIDELINES:
- For mutually exclusive events (only one option wins — e.g. an election,
  a championship, "which date will X happen"), your probabilities should
  sum close to 1.0 across all outcomes. Most options will be small (1/N
  base rate) unless you have strong evidence one is the favorite.
- For non-exclusive events (each outcome is independent — e.g. "will X
  be mentioned on the show?"), probabilities can sum to any value.
- The MARKET PRICES shown are a strong prior. Deviate only when you can
  articulate specific evidence the market has missed.
- Consider base rates, recency of evidence, and avoid overconfidence.

Output JSON with this exact shape:
    {
        "exclusive": true|false,
        "outcomes": {
            "<market_ticker_1>": <p_yes 0.01-0.99>,
            "<market_ticker_2>": <p_yes 0.01-0.99>,
            ...
        },
        "rationale": "<2-3 sentences>"
    }
Do not include any other text outside the JSON object.
"""


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    return OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")


def _format_market_block(idx: int, ticker: str, market_meta: dict, market_info: dict | None) -> str:
    parts = [f"({idx}) {ticker}: {market_meta.get('title') or market_meta.get('subtitle') or ticker}"]
    if market_meta.get("subtitle"):
        parts.append(f"    subtitle: {market_meta['subtitle']}")
    if market_meta.get("yes_sub_title"):
        parts.append(f"    YES means: {market_meta['yes_sub_title']}")
    if market_meta.get("rules") or market_meta.get("rules_primary"):
        rules = market_meta.get("rules") or market_meta.get("rules_primary")
        parts.append(f"    rules: {rules[:200]}")
    if market_info:
        yes_ask = market_info.get("yes_ask")
        no_ask = market_info.get("no_ask")
        if yes_ask is not None and no_ask is not None and (yes_ask + no_ask) > 0:
            mid = (yes_ask + (100 - no_ask)) / 200
            parts.append(f"    market price: YES ask {yes_ask}¢, NO ask {no_ask}¢  (implied P(YES) ≈ {mid:.3f})")
        elif market_info.get("last_price") is not None:
            parts.append(f"    market last price: {market_info['last_price']}¢")
    return "\n".join(parts)


def _build_event_prompt(
    event_title: str,
    event_category: str,
    siblings: list[tuple[str, dict, dict | None]],
    close_time: str | None,
) -> str:
    parts = [
        f"EVENT: {event_title}",
        f"Category: {event_category}",
    ]
    if close_time:
        parts.append(f"Close time: {close_time}")
    parts.append(f"\nThis event has {len(siblings)} constituent binary markets:")
    for i, (ticker, meta, mi) in enumerate(siblings, 1):
        parts.append(_format_market_block(i, ticker, meta, mi))
    parts.append(
        "\nRespond with a JSON object as specified in the system prompt. "
        "Use the EXACT market_ticker strings shown above as keys in `outcomes`."
    )
    return "\n".join(parts)


def _ask_event(
    event_title: str,
    event_category: str,
    siblings: list[tuple[str, dict, dict | None]],
    close_time: str | None,
    retries: int = 2,
) -> dict | None:
    """One LLM call covering all siblings. Returns the parsed JSON or None."""
    model = os.environ.get("OPENROUTER_MODEL", "x-ai/grok-4.20")
    temperature = float(os.environ.get("OPENROUTER_TEMPERATURE", "0.5"))
    max_tokens = int(os.environ.get("OPENROUTER_MAX_TOKENS", "600"))

    prompt = _build_event_prompt(event_title, event_category, siblings, close_time)

    for attempt in range(retries + 1):
        try:
            resp = _client().chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            )
            text = (resp.choices[0].message.content or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
            data = json.loads(text)
            if "outcomes" not in data or not isinstance(data["outcomes"], dict):
                raise ValueError("missing/invalid 'outcomes' field")
            return data
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"[event_aware] attempt {attempt+1}: {type(e).__name__}: {e}\n")
            time.sleep(0.5 * (attempt + 1))
    return None


def predict_event(
    event_title: str,
    event_category: str,
    siblings: list[tuple[str, dict, dict | None]],
    close_time: str | None = None,
) -> dict[str, float]:
    """Predict for every market in an event. Returns {ticker: p_yes}.

    `siblings` is a list of (market_ticker, market_meta, market_info)
    tuples for ALL markets sharing the event.

    If the LLM marks the event as `exclusive` and probabilities don't
    sum close to 1, we renormalize. For non-exclusive events we just
    clamp.
    """
    result = _ask_event(event_title, event_category, siblings, close_time)
    if not result:
        # Fallback: per-ticker market price (or 0.5)
        out: dict[str, float] = {}
        for ticker, _meta, mi in siblings:
            p = 0.5
            if mi:
                if mi.get("yes_ask") is not None and mi.get("no_ask") is not None:
                    p = (mi["yes_ask"] + (100 - mi["no_ask"])) / 200
                elif mi.get("last_price") is not None:
                    p = mi["last_price"] / 100
            out[ticker] = max(0.01, min(0.99, p))
        return out

    raw = {}
    for ticker, _, _ in siblings:
        v = result["outcomes"].get(ticker)
        if v is None:
            raw[ticker] = 0.5
        else:
            try:
                raw[ticker] = max(0.01, min(0.99, float(v)))
            except (TypeError, ValueError):
                raw[ticker] = 0.5

    if result.get("exclusive") and len(raw) > 1:
        total = sum(raw.values())
        if total > 0:
            raw = {t: max(0.01, min(0.99, v / total)) for t, v in raw.items()}

    return raw


def predict(
    event: dict,
    market_info: dict | None = None,
    candidate_set: list | None = None,
) -> dict:
    """Single-market entry point. Uses event-aware prompting if `candidate_set`
    is provided (list of Samples sharing the same event_ticker); otherwise
    falls back to the zero-shot path.
    """
    et = event.get("event_ticker") or ""
    requested_ticker = event.get("market_ticker") or et

    siblings = []
    if candidate_set:
        for s in candidate_set:
            s_et = (s.event.get("event_ticker") or "")
            if s_et == et:
                siblings.append((
                    s.event.get("market_ticker") or "",
                    s.event,
                    s.market_info,
                ))
        # Ensure the requested market is in there
        if not any(t == requested_ticker for t, _, _ in siblings):
            siblings.append((requested_ticker, event, market_info))
    else:
        siblings = [(requested_ticker, event, market_info)]

    if len(siblings) == 1:
        # Single-market fallback — delegate to the zero-shot predictor with
        # market context, which is what we want here.
        from .openrouter_zero_shot import predict as zs_predict
        return zs_predict(event, market_info)

    preds = predict_event(
        event.get("title", ""),
        event.get("category", ""),
        siblings,
        close_time=str(event.get("close_time") or ""),
    )
    p = preds.get(requested_ticker, 0.5)
    return {
        "p_yes": p,
        "rationale": f"event_aware on event {et} ({len(siblings)} siblings) → {p:.3f}",
    }
