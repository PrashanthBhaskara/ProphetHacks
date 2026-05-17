"""Prophet Hacks forecast-track prediction server.

Exposes p_yes per Kalshi binary market. Designed to be polled by judges
during the 2-week evaluation window after the May 17, 2026 submission.

Endpoints:
  GET  /                          -> usage + version
  GET  /healthz                   -> liveness probe (Render health check)
  GET  /predict?ticker=KX...      -> one prediction
  POST /predict                   -> {"tickers": ["KX...", ...]} batch
  POST /predict_event             -> reference-predictor-shaped payload
                                     {"market_ticker": "...", "title": "...", ...}

Model: market mid-price from Kalshi /markets/{ticker}, optionally
shrunk through global Platt if server/calibration.json exists.
Per our audit, this is within ~0.02 Brier of every other calibration
we tested on 2026 Sports-heavy data — a defensible floor.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from . import calibration, kalshi
from .kalshi import get_event_markets

VERSION = "1.0.0"
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "60"))
MAX_BATCH = int(os.environ.get("MAX_BATCH", "200"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("prophet-server")

app = FastAPI(title="Prophet Hacks forecast API", version=VERSION)

_cache: dict[str, tuple[float, dict]] = {}


def _cache_get(ticker: str) -> dict | None:
    hit = _cache.get(ticker)
    if not hit:
        return None
    ts, payload = hit
    if time.time() - ts > CACHE_TTL_SECONDS:
        _cache.pop(ticker, None)
        return None
    return payload


def _cache_put(ticker: str, payload: dict) -> None:
    _cache[ticker] = (time.time(), payload)


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _market_implied_p(market: dict) -> tuple[float, str]:
    """Return (p_yes, source_label).

    Kalshi returns prices in two shapes depending on the market:
      - cents ints: yes_ask=42, no_ask=58
      - dollar strings: yes_ask_dollars="0.42", no_ask_dollars="0.58"
    We normalize both to dollars in [0, 1] before averaging.
    """
    yes_ask = _to_float(market.get("yes_ask_dollars"))
    no_ask = _to_float(market.get("no_ask_dollars"))
    if yes_ask is None or no_ask is None:
        yc = _to_float(market.get("yes_ask"))
        nc = _to_float(market.get("no_ask"))
        if yc is not None and nc is not None:
            yes_ask, no_ask = yc / 100, nc / 100
    if yes_ask is not None and no_ask is not None and (yes_ask + no_ask) > 0:
        return (yes_ask + (1 - no_ask)) / 2, "mid"
    last = _to_float(market.get("last_price_dollars"))
    if last is None:
        lc = _to_float(market.get("last_price"))
        if lc is not None:
            last = lc / 100
    if last is not None and last > 0:
        return last, "last"
    return 0.5, "fallback"


def _predict_one(ticker: str) -> dict:
    cached = _cache_get(ticker)
    if cached is not None:
        return {**cached, "cached": True}

    market = kalshi.get_market(ticker)
    if market is None:
        payload = {
            "market_ticker": ticker,
            "p_yes": 0.5,
            "source": "unknown_ticker",
            "calibration": calibration.info()["mode"],
            "ts": int(time.time()),
        }
        _cache_put(ticker, payload)
        return {**payload, "cached": False}

    raw_p, source = _market_implied_p(market)
    p_yes = calibration.apply(raw_p)
    payload = {
        "market_ticker": ticker,
        "p_yes": round(p_yes, 6),
        "raw_market_p": round(raw_p, 6),
        "source": source,
        "calibration": calibration.info()["mode"],
        "status": market.get("status"),
        "close_time": market.get("close_time"),
        "ts": int(time.time()),
    }
    _cache_put(ticker, payload)
    return {**payload, "cached": False}


class BatchRequest(BaseModel):
    tickers: list[str] = Field(..., max_length=MAX_BATCH)


class EventRequest(BaseModel):
    market_ticker: str
    title: str | None = None
    rules: str | None = None
    category: str | None = None
    close_time: str | None = None


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "prophet-hacks-forecast",
        "version": VERSION,
        "calibration": calibration.info(),
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "endpoints": [
            "GET  /healthz",
            "GET  /predict?ticker=KX...",
            "POST /predict          {tickers: [...]}",
            "POST /predict_event    {market_ticker, title?, rules?, ...}",
        ],
    }


@app.get("/healthz")
@app.get("/health")
def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "version": VERSION,
        "calibration": calibration.info(),
        "cache_size": len(_cache),
    }


@app.get("/predict")
def predict_get(ticker: str = Query(..., min_length=3, max_length=128)) -> dict:
    out = _predict_one(ticker)
    log.info("predict ticker=%s p_yes=%.4f source=%s cached=%s",
             ticker, out["p_yes"], out.get("source"), out["cached"])
    return out


_TICKER_KEYS = (
    "market_ticker", "ticker", "market", "event_ticker",
    "marketTicker", "eventTicker", "market_name", "marketName",
    "name", "id", "market_id", "marketId",
)
_LIST_KEYS = (
    "tickers", "market_tickers", "markets", "events", "event_tickers",
    "market_names", "marketNames", "marketTickers", "eventTickers",
    "probabilities", "predictions", "candidate_set", "candidates",
)


def _extract_tickers(payload: Any) -> list[str]:
    """Pull tickers out of an arbitrary POST body.

    Judges may send any of: {"tickers": [...]}, {"market_ticker": "..."},
    {"event": {"markets": [{"ticker": "..."}]}}, a bare list, etc.
    We never raise — return [] and let the caller decide what to do.
    """
    seen: set[str] = set()
    out: list[str] = []

    def add(v: Any) -> None:
        if isinstance(v, str) and v.strip() and v not in seen:
            seen.add(v)
            out.append(v.strip())

    def walk(node: Any) -> None:
        if isinstance(node, str):
            add(node)
            return
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if isinstance(node, dict):
            for k in _TICKER_KEYS:
                if k in node:
                    add(node[k]) if isinstance(node[k], str) else walk(node[k])
            for k in _LIST_KEYS:
                if k in node:
                    walk(node[k])
            for k in ("event", "data", "payload", "body", "request"):
                if k in node:
                    walk(node[k])
        # ignore other types

    walk(payload)
    return out[:MAX_BATCH]


def _predict_outcomes(event_ticker: str, outcomes: list[str]) -> list[dict]:
    """Look up an event's markets and return per-outcome-label probabilities.

    Matches each outcome label to the market whose yes_sub_title contains it
    (case-insensitive). Falls back to 1/N uniform if no match found.
    """
    markets = get_event_markets(event_ticker)
    n = max(1, len(outcomes))
    uniform = round(1.0 / n, 6)

    results = []
    for outcome in outcomes:
        outcome_lower = outcome.lower()
        matched = None
        for m in markets:
            sub = (m.get("yes_sub_title") or m.get("subtitle") or "").lower()
            title = (m.get("title") or "").lower()
            if outcome_lower in sub or outcome_lower in title:
                matched = m
                break

        if matched is not None:
            raw_p, source = _market_implied_p(matched)
            p = calibration.apply(raw_p)
            results.append({
                "market": outcome,
                "probability": round(p, 6),
                "market_ticker": matched.get("ticker"),
                "source": source,
                "status": matched.get("status"),
                "ts": int(time.time()),
                "cached": False,
            })
        else:
            results.append({
                "market": outcome,
                "probability": uniform,
                "source": "no_match",
                "ts": int(time.time()),
                "cached": False,
            })

    # Normalize so probabilities sum to 1
    total = sum(r["probability"] for r in results)
    if total > 0:
        for r in results:
            r["probability"] = round(r["probability"] / total, 6)

    return results


@app.post("/predict")
async def predict_post(request: Request) -> dict[str, Any]:
    """Maximally tolerant POST endpoint.

    Accepts any JSON shape, extracts ticker(s) from common field names,
    returns predictions. Falls back to an empty-tickers diagnostic
    response instead of 422 so clients always see a 200.
    """
    try:
        body = await request.json()
    except Exception:
        body = None

    if not body:
        log.warning("predict_post empty body")
        return {
            "rationale": "no market identifiers found in request body; returning uniform fallback",
            "probabilities": [{"market": "fallback", "probability": 0.5}],
            "predictions": [],
            "p_yes": 0.5,
            "ts": int(time.time()),
        }

    # Multi-outcome path: body has an outcomes list — look up the event and
    # match each outcome label to its Kalshi market by yes_sub_title.
    outcomes = body.get("outcomes") if isinstance(body, dict) else None
    event_ticker = (
        (body.get("event_ticker") or body.get("market_ticker"))
        if isinstance(body, dict) else None
    )
    if outcomes and len(outcomes) > 0 and event_ticker:
        outcome_results = _predict_outcomes(event_ticker, outcomes)
        sources = list({r.get("source", "?") for r in outcome_results})
        log.info("predict_post event=%s outcomes=%d sources=%s", event_ticker, len(outcome_results), sources)
        return {
            "rationale": f"Per-outcome Kalshi market mid-price for {event_ticker}.",
            "probabilities": [{"market": r["market"], "probability": r["probability"]} for r in outcome_results],
            "predictions": outcome_results,
            "ts": int(time.time()),
        }

    # Single-ticker fallback path
    tickers = _extract_tickers(body)
    if not tickers:
        log.warning("predict_post no tickers found in body=%r", body)
        return {
            "rationale": "no market identifiers found in request body; returning uniform fallback",
            "probabilities": [{"market": "fallback", "probability": 0.5}],
            "predictions": [],
            "p_yes": 0.5,
            "ts": int(time.time()),
        }

    results = [_predict_one(t) for t in tickers]
    log.info("predict_post n=%d first=%s p_yes=%.4f",
             len(results), tickers[0], results[0].get("p_yes", -1))

    probabilities = [
        {"market": r["market_ticker"], "probability": round(float(r["p_yes"]), 6)}
        for r in results
    ]

    response: dict[str, Any] = {
        "rationale": f"Probabilities derived from Kalshi market mid-price across {len(probabilities)} outcome(s).",
        "probabilities": probabilities,
        "predictions": results,
        "ts": int(time.time()),
    }
    if len(results) == 1:
        r = results[0]
        response["market_ticker"] = r["market_ticker"]
        response["p_yes"] = r["p_yes"]
        response["probability"] = r["p_yes"]
    return response


@app.post("/predict_event")
def predict_event(req: EventRequest) -> dict:
    out = _predict_one(req.market_ticker)
    if req.title:
        out["title"] = req.title
    log.info("predict_event ticker=%s title=%s p_yes=%.4f",
             req.market_ticker, (req.title or "")[:40], out["p_yes"])
    return out
