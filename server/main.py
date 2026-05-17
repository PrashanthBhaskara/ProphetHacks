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

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from . import calibration, kalshi

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


@app.post("/predict")
def predict_batch(req: BatchRequest) -> dict[str, Any]:
    if not req.tickers:
        raise HTTPException(status_code=400, detail="tickers must be non-empty")
    if len(req.tickers) > MAX_BATCH:
        raise HTTPException(status_code=400, detail=f"max batch size is {MAX_BATCH}")
    results = [_predict_one(t) for t in req.tickers]
    log.info("predict_batch n=%d", len(results))
    return {"predictions": results, "ts": int(time.time())}


@app.post("/predict_event")
def predict_event(req: EventRequest) -> dict:
    out = _predict_one(req.market_ticker)
    if req.title:
        out["title"] = req.title
    log.info("predict_event ticker=%s title=%s p_yes=%.4f",
             req.market_ticker, (req.title or "")[:40], out["p_yes"])
    return out
