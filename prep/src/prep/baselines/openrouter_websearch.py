"""Grok with OpenRouter web search plugin enabled.

This is the live-submission predictor: Grok gets to search the web and X to
gather pre-resolution information that the market may not have priced.

**Contamination caveat for offline backtest:** Our 2026 events have already
resolved, so a naive web search returns post-resolution articles. To probe
the methodology on past events, we explicitly tell Grok in the system prompt
that "today is [snapshot_date]" and forbid using post-snapshot information.
We then INSPECT the returned rationale to check whether Grok cites future
articles. If it does, the offline Brier is contaminated. For the LIVE
submission (May 17-27 forward), this concern disappears.

Env:
    OPENROUTER_API_KEY     required
    OPENROUTER_MODEL       default x-ai/grok-4.3
    OPENROUTER_WEB_RESULTS default 5  (number of web results to retrieve)
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from functools import lru_cache

import requests

SYSTEM_TEMPLATE = """\
You are a forecaster evaluating a binary prediction market on Kalshi.

**TODAY'S DATE FOR THIS EVALUATION: {snapshot_date}**

You have web search available. CRITICAL RULES:
1. Treat the date above as "today". You MUST NOT use any information that
   was published AFTER that date. If a search result is dated after that day,
   ignore it completely.
2. Your rationale must cite specific sources you used, with their publication
   date. If you can't find a source dated before "today", say so explicitly
   and fall back to the market price.

The market price has been set by many traders and is a strong prior. Deviate
from it only when web search reveals specific information dated BEFORE
{snapshot_date} that the market appears to have missed.

Respond with ONLY valid JSON:
{{"p_yes": <float 0.01-0.99>, "deviated": <bool>, "sources_used": ["<source 1 with date>", ...], "rationale": "<2-3 sentences>"}}
If you couldn't find pre-snapshot evidence, set deviated=false and output the market_mid exactly.
"""


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
    if mid is not None:
        parts.append("")
        parts.append(f"MARKET STATE: market_mid (default answer if no edge): {mid:.3f}")
        if market_info.get("yes_ask") is not None:
            parts.append(f"  YES ask: {market_info['yes_ask']}¢   NO ask: {market_info.get('no_ask','?')}¢")
    parts.append("")
    parts.append("Search the web for pre-snapshot-date information that the market may have missed.")
    parts.append("Output JSON as specified.")
    return "\n".join(parts)


def predict(
    event: dict,
    market_info: dict | None = None,
    *,
    snapshot_date: str | None = None,
) -> dict:
    """Web-search-augmented Grok forecast.

    `snapshot_date` is a YYYY-MM-DD string. If not provided, uses today.
    For backtest: pass the sample's snapshot_time to constrain Grok to
    pre-snapshot info (probe methodology).
    For live: omit or set to today.
    """
    mid = _market_mid(market_info)
    if snapshot_date is None:
        from datetime import datetime, timezone
        snapshot_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    else:
        snapshot_date = str(snapshot_date)[:10]

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    model = os.environ.get("OPENROUTER_MODEL", "x-ai/grok-4.3")
    web_results = int(os.environ.get("OPENROUTER_WEB_RESULTS", "5"))

    sys_msg = SYSTEM_TEMPLATE.format(snapshot_date=snapshot_date)
    user_msg = _build_prompt(event, market_info)

    payload = {
        "model": model,
        "max_tokens": 800,
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ],
        "plugins": [{"id": "web", "max_results": web_results}],
    }

    last_err = None
    for attempt in range(3):
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json=payload,
                timeout=120,
            )
            data = r.json()
            if "choices" not in data:
                last_err = f"no choices: {data}"
                time.sleep(0.5 * (attempt + 1))
                continue
            text = (data["choices"][0]["message"]["content"] or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
            parsed = json.loads(text)
            p_raw = max(0.01, min(0.99, float(parsed["p_yes"])))
            cost = data.get("usage", {}).get("cost", 0.0)
            return {
                "p_yes": p_raw,
                "deviated": bool(parsed.get("deviated", False)),
                "sources_used": parsed.get("sources_used", []),
                "rationale": parsed.get("rationale", ""),
                "market_mid": mid,
                "cost_usd": cost,
                "snapshot_date": snapshot_date,
            }
        except Exception as e:
            last_err = e
            time.sleep(1.0 * (attempt + 1))
    sys.stderr.write(f"[websearch] failed after 3 attempts: {last_err}\n")
    return {"p_yes": mid if mid is not None else 0.5,
            "rationale": f"fallback: {last_err}",
            "market_mid": mid, "deviated": False, "cost_usd": 0.0}
