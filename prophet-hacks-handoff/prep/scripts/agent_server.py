"""FastAPI agent server for the Prophet Arena forecasting track.

Matches the wire contract from https://prophetarena.co/developer :
  - POST /predict accepts an Event JSON body
  - Returns {"probabilities": [{"market": <outcome_label>, "probability": <float>}, ...]}

Behavior:
  - Loads `config/ensemble.example.json` (or path from PROPHET_CONFIG env var)
  - Runs every enabled forecaster lane in parallel via ThreadPoolExecutor
  - Aggregates with the existing ensemble + calibration stack
  - Returns the calibrated distribution, mapping back onto the event's `outcomes`

Run locally:
    uvicorn scripts.agent_server:app --host 0.0.0.0 --port 8000

Then point the CLI at it:
    prophet forecast predict --events events.json --agent-url http://localhost:8000/predict
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
)
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prep.calibration import CalibrationConfig  # noqa: E402
from prep.ensemble import JudgeConfig, aggregate_forecasts, forecast_members_parallel  # noqa: E402
from prep.forecasters import ForecasterConfig  # noqa: E402
from prep.kalshi import get_market, list_markets  # noqa: E402
from prep.packets import packet_from_arena_event  # noqa: E402
from prep.schemas import KalshiQuote, MarketPacket  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "FINAL.json"
CONFIG_PATH = Path(os.environ.get("PROPHET_CONFIG", DEFAULT_CONFIG_PATH))

# Wall-clock budget for the entire /predict call (lane fan-out + aggregation +
# judge). 9m30s — leaves ~90s of slack over the grok lane's 8-minute budget so
# a stuck grok call can fall back cleanly before the outer deadline fires.
# On timeout we return market price: market_mid for binary Kalshi events,
# uniform across `outcomes` otherwise. Override with ENSEMBLE_TIMEOUT_SECONDS.
DEFAULT_ENSEMBLE_TIMEOUT_SECONDS = 570.0


# --- Wire schema (Prophet Arena dev docs) ---------------------------------

class ArenaEvent(BaseModel):
    """Inbound Event body. Extra fields ignored — Arena adds metadata over time."""

    event_ticker: str | None = None
    market_ticker: str | None = None
    task_id: str | None = None  # newer dataset format may send this
    title: str
    subtitle: str | None = None
    description: str | None = None
    context: str | None = None  # newer dataset rename
    category: str | None = None
    rules: str | None = None
    close_time: str | None = None
    predict_by: str | None = None  # newer dataset rename
    outcomes: list[str]
    resolved_outcome: dict | None = None  # always None for live predict, present in historical

    model_config = {"extra": "allow"}


class OutcomeProbability(BaseModel):
    market: str
    probability: float


class PredictionResponse(BaseModel):
    probabilities: list[OutcomeProbability]


# --- Config loading -------------------------------------------------------

def _load_models() -> tuple[list[ForecasterConfig], CalibrationConfig, float, JudgeConfig]:
    cfg = json.loads(CONFIG_PATH.read_text())
    models = [
        ForecasterConfig.from_dict(m)
        for m in cfg.get("models", [])
        if m.get("enabled", True)
    ]
    calibration = CalibrationConfig.from_dict(cfg.get("calibration"))
    ensemble_cfg = cfg.get("ensemble", {})
    market_anchor_weight = float(ensemble_cfg.get("market_anchor_weight", 1.5))
    judge = JudgeConfig.from_dict(ensemble_cfg.get("judge"))
    return models, calibration, market_anchor_weight, judge


_models: list[ForecasterConfig] = []
_calibration: CalibrationConfig = CalibrationConfig()
_market_anchor_weight: float = 1.5
_judge: JudgeConfig = JudgeConfig()


# --- FastAPI app ----------------------------------------------------------

app = FastAPI(title="ProphetHacks Forecast Agent")


@app.on_event("startup")
def _startup() -> None:
    global _models, _calibration, _market_anchor_weight, _judge
    _models, _calibration, _market_anchor_weight, _judge = _load_models()
    logger.info("Loaded %d enabled lanes: %s", len(_models), [m.name for m in _models])


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "enabled_lanes": [m.name for m in _models],
        "judge_enabled": _judge.enabled,
        "judge_model": _judge.model if _judge.enabled else None,
        "config": str(CONFIG_PATH),
    }


def _f(val) -> float | None:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _market_mid(m: dict) -> float | None:
    bid = _f(m.get("yes_bid_dollars"))
    ask = _f(m.get("yes_ask_dollars"))
    last = _f(m.get("last_price_dollars"))
    if bid is not None and ask is not None and (bid > 0 or ask > 0):
        return (bid + ask) / 2.0
    if last is not None and last > 0:
        return last
    return None


def _match_market(outcome: str, markets: list[dict]) -> dict | None:
    """Match an outcome label to its Kalshi market by searching title/subtitle."""
    outcome_lower = outcome.lower()
    for m in markets:
        title = (m.get("title") or "").lower()
        sub = (m.get("yes_sub_title") or m.get("subtitle") or "").lower()
        if outcome_lower in title or outcome_lower in sub:
            return m
    return None


def _enrich_packet(packet: MarketPacket) -> MarketPacket:
    """Fetch live Kalshi data and inject it into the packet before forecasting.

    Binary events: fetch the single market by market_ticker.
    Multi-outcome events: fetch all markets under the event_ticker and build
    a market_implied_probabilities dict in packet.retrieval.
    """
    market_ticker = packet.market_ticker
    event_ticker = packet.event_ticker

    try:
        # --- Binary: try fetching by market_ticker first ---
        if market_ticker:
            market = get_market(market_ticker)
            if market:
                packet.kalshi = KalshiQuote(
                    yes_bid=_f(market.get("yes_bid_dollars")),
                    yes_ask=_f(market.get("yes_ask_dollars")),
                    no_bid=_f(market.get("no_bid_dollars")),
                    no_ask=_f(market.get("no_ask_dollars")),
                    last_price=_f(market.get("last_price_dollars")),
                    volume=_f(market.get("volume_fp")),
                    open_interest=_f(market.get("open_interest_fp")),
                    yes_bid_size=_f(market.get("yes_bid_size_fp")),
                    yes_ask_size=_f(market.get("yes_ask_size_fp")),
                )
                if market.get("rules_primary") and not packet.retrieval.get("description"):
                    packet.retrieval["description"] = market["rules_primary"]
                if market.get("yes_sub_title") and not packet.subtitle:
                    packet.subtitle = market["yes_sub_title"]
                logger.info(
                    "Enriched binary %s: mid=%.3f vol=%s oi=%s",
                    market_ticker, packet.kalshi.market_mid,
                    market.get("volume_fp"), market.get("open_interest_fp"),
                )
                return packet

        # --- Multi-outcome: fetch all markets under the event ---
        ticker = event_ticker or market_ticker
        if not ticker:
            return packet

        markets = list_markets(event_ticker=ticker, status=None, limit=100)
        if not markets:
            logger.warning("Kalshi returned no data for ticker %s", ticker)
            return packet

        # Build market_implied_probabilities keyed by outcome label
        implied: dict[str, float] = {}
        market_data: list[dict] = []
        for outcome in packet.outcomes:
            m = _match_market(outcome, markets)
            if m is None:
                continue
            mid = _market_mid(m)
            if mid is not None:
                implied[outcome] = round(mid, 4)
            market_data.append({
                "outcome": outcome,
                "ticker": m.get("ticker"),
                "yes_bid": _f(m.get("yes_bid_dollars")),
                "yes_ask": _f(m.get("yes_ask_dollars")),
                "mid": mid,
                "volume": _f(m.get("volume_fp")),
                "open_interest": _f(m.get("open_interest_fp")),
            })

        if implied:
            packet.retrieval["market_implied_probabilities"] = implied
        if market_data:
            packet.retrieval["market_data"] = market_data

        logger.info(
            "Enriched multi-outcome %s: %d/%d outcomes matched, implied=%s",
            ticker, len(implied), len(packet.outcomes),
            {k: f"{v:.3f}" for k, v in implied.items()},
        )

    except Exception as exc:  # noqa: BLE001
        logger.warning("Kalshi enrichment failed for %s: %s", market_ticker or event_ticker, exc)

    return packet


def _market_price_response(packet, outcomes: list[str]) -> PredictionResponse:
    """Build a market-price response mapped onto `outcomes`.

    Used as the timeout/error fallback. Binary YES/NO events with a Kalshi
    quote return market_mid; everything else returns uniform across outcomes
    (best we can do when no market price is available).
    """
    kalshi = getattr(packet, "kalshi", None)
    if tuple(outcomes) == ("YES", "NO") and kalshi is not None:
        try:
            mid = float(kalshi.market_mid)
        except (TypeError, ValueError, AttributeError):
            mid = 0.5
        dist = {"YES": mid, "NO": 1.0 - mid}
    else:
        share = 1.0 / max(1, len(outcomes))
        dist = {o: share for o in outcomes}
    return PredictionResponse(
        probabilities=[
            OutcomeProbability(market=o, probability=float(dist.get(o, 0.0)))
            for o in outcomes
        ]
    )


def _compute_ensemble(event: ArenaEvent, deadline: float) -> PredictionResponse:
    """Inner ensemble computation. Caller enforces the wall-clock deadline.

    Lanes that haven't completed by the deadline are dropped from the
    aggregate; aggregation still runs against whatever finished in time.
    Returns the calibrated PredictionResponse mapped onto event.outcomes.
    """
    packet = packet_from_arena_event(event.model_dump())
    packet = _enrich_packet(packet)

    if not _models:
        # No lanes enabled — uniform over outcomes is the best we can do.
        share = 1.0 / max(1, len(event.outcomes))
        return PredictionResponse(
            probabilities=[OutcomeProbability(market=o, probability=share) for o in event.outcomes]
        )

    # Inner parallel exec is delegated to forecast_members_parallel. Per-lane
    # 8-minute budgets live in forecasters/base.py's forecast_from_config; the
    # outer 9m30s budget is enforced by predict_endpoint via fut.result(timeout=).
    run = forecast_members_parallel(_models, packet, continue_on_error=True)
    forecasts = run.members
    errors = run.errors
    for error in errors:
        logger.warning("lane failed: %s", error)

    if not forecasts:
        # All lanes failed or timed out. Don't 502 the eval — degrade to the
        # anchor distribution that the empty-members path in aggregate_forecasts
        # already produces (market_mid for binary Kalshi, uniform otherwise).
        logger.warning("all lanes failed for %s: %s", packet.market_ticker, "; ".join(errors))

    supervisor = aggregate_forecasts(
        packet,
        forecasts,
        calibration=_calibration,
        market_anchor_weight=_market_anchor_weight,
        judge=_judge,
    )

    # Map calibrated distribution onto the event's outcomes exactly (preserve order).
    dist = supervisor.calibrated_probabilities
    return PredictionResponse(
        probabilities=[
            OutcomeProbability(market=o, probability=float(dist.get(o, 0.0)))
            for o in event.outcomes
        ]
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: Request) -> PredictionResponse:
    """Wall-clock-bounded /predict.

    Accepts either a single Event object or a list containing one Event object.
    Runs the full ensemble (lane fan-out + calibration + judge) inside a
    deadline. If the whole flow hasn't returned within ENSEMBLE_TIMEOUT_SECONDS
    (default 570s), we abandon it and return market price.

    Also callable in-process as `predict(ArenaEvent(**event_dict))` for local
    testing via `prophet forecast predict --local scripts.agent_server`.
    """
    global _models, _calibration, _market_anchor_weight, _judge
    if not _models:
        _models, _calibration, _market_anchor_weight, _judge = _load_models()

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    if isinstance(body, list):
        if not body:
            raise HTTPException(status_code=422, detail="Event list is empty")
        event_data = body[0]
    else:
        event_data = body

    try:
        event = ArenaEvent.model_validate(event_data)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    budget = float(os.environ.get("ENSEMBLE_TIMEOUT_SECONDS", DEFAULT_ENSEMBLE_TIMEOUT_SECONDS))
    deadline = time.monotonic() + budget

    supervisor_pool = ThreadPoolExecutor(max_workers=1)
    try:
        fut = supervisor_pool.submit(_compute_ensemble, event, deadline)
        try:
            return fut.result(timeout=budget)
        except FuturesTimeoutError:
            packet = packet_from_arena_event(event.model_dump())
            logger.warning(
                "ensemble exceeded %.0fs budget for %s; returning market price",
                budget, packet.market_ticker,
            )
            return _market_price_response(packet, event.outcomes)
        except Exception as exc:  # noqa: BLE001
            packet = packet_from_arena_event(event.model_dump())
            logger.exception(
                "ensemble raised %s for %s; returning market price",
                type(exc).__name__, packet.market_ticker,
            )
            return _market_price_response(packet, event.outcomes)
    finally:
        supervisor_pool.shutdown(wait=False, cancel_futures=True)
